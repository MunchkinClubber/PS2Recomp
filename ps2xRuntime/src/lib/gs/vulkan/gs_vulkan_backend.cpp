// Vulkan GS backend (see gs_vulkan_backend.h).
//
// GS memory (the 4 MiB VRAM array) stays the reference copy for everything that is not a draw:
// transfers, CLUT loads, readbacks and presentation run on a synchronous GSCpuBackend over the
// same memory. Draws go to GPU render targets, one per (FBP, FBW, PSM) frame buffer and per Z
// buffer, and two rules keep the two copies coherent:
//   - a target holds newer data than GS memory in its "dirty" rectangle; before anything reads
//     or writes those pages on the CPU side (texture decode, transfer, CLUT, readback, a draw into
//     an overlapping target) the rectangle is downloaded into GS memory;
//   - a target's rows are "stale" when GS memory changed under them (a CPU-side write, or a draw
//     into an overlapping target); they are uploaded again before the target is next drawn to.
// Textures are decoded from GS memory (after downloading any target that overlaps them) into
// RGBA8 images, cached until one of their pages is written.
//
// The shaders reproduce the CPU renderer's arithmetic (see shaders/gs.frag), so the two agree up
// to GPU interpolation and blending rounding. Draw states the GPU path does not cover (lines,
// points, destination-alpha test, blends with a factor above one, ...) are drawn by the CPU
// renderer instead, with the same coherency rules.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "volk.h"

#include "runtime/gs/gs_vulkan_backend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "shaders/gs_shaders.h"
#if defined(PS2X_HOST_SDL3)
#include "ps2_host_vulkan.h"
#endif
struct HostVulkanShared;
struct HostGpuFrame;

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <cfenv>
#include <cstddef>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>

namespace
{
    // Set of the 512 GS memory pages (8 KiB each).
    struct PageSet
    {
        uint64_t w[8] = {};
        void set(uint32_t p) { w[(p >> 6) & 7u] |= 1ull << (p & 63u); }
        void set() { for (uint64_t &x : w) x = ~0ull; }
        bool test(uint32_t p) const { return (w[(p >> 6) & 7u] >> (p & 63u)) & 1u; }
        void reset() { for (uint64_t &x : w) x = 0; }
        bool any() const { uint64_t a = 0; for (uint64_t x : w) a |= x; return a != 0; }
        PageSet operator&(const PageSet &o) const { PageSet r; for (int i = 0; i < 8; ++i) r.w[i] = w[i] & o.w[i]; return r; }
        PageSet operator|(const PageSet &o) const { PageSet r; for (int i = 0; i < 8; ++i) r.w[i] = w[i] | o.w[i]; return r; }
        PageSet operator~() const { PageSet r; for (int i = 0; i < 8; ++i) r.w[i] = ~w[i]; return r; }
        PageSet &operator&=(const PageSet &o) { for (int i = 0; i < 8; ++i) w[i] &= o.w[i]; return *this; }
        PageSet &operator|=(const PageSet &o) { for (int i = 0; i < 8; ++i) w[i] |= o.w[i]; return *this; }
        bool intersects(const PageSet &o) const { uint64_t a = 0; for (int i = 0; i < 8; ++i) a |= w[i] & o.w[i]; return a != 0; }
        // Calls fn(page) for every page in the set, in order.
        template <typename F>
        void forEach(F &&fn) const
        {
            for (uint32_t i = 0; i < 8u; ++i)
                for (uint64_t x = w[i]; x; x &= x - 1u)
                {
                    unsigned long b;
#if defined(_MSC_VER)
                    _BitScanForward64(&b, x);
#else
                    b = static_cast<unsigned long>(__builtin_ctzll(x));
#endif
                    fn(i * 64u + static_cast<uint32_t>(b));
                }
        }
    };
    constexpr uint32_t kTargetHeight = 1024u;
    constexpr uint32_t kRingSize = 64u << 20; // two halves: one per command buffer in flight
    constexpr uint32_t kRingHalf = kRingSize / 2u;
    constexpr uint32_t kCompSets = 2048u;
    constexpr uint32_t kMaxBatchVertices = 3u * 20000u;
    constexpr size_t kMaxTextures = 1536u;

    bool envFlag(const char *name)
    {
        const char *v = std::getenv(name);
        return v && *v && *v != '0';
    }

    struct PageDims
    {
        uint32_t w, h;
    };

    PageDims pageDims(uint8_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            return {64u, 64u};
        case GS_PSM_T8:
            return {128u, 64u};
        case GS_PSM_T4:
            return {128u, 128u};
        default:
            return {64u, 32u};
        }
    }

    // Pages (8 KiB) touched by the inclusive pixel rectangle of a buffer at block `bp`.
    void addRectPages(PageSet &set, uint32_t bp, uint32_t bw, uint8_t psm, int x0, int y0, int x1, int y1)
    {
        if (x1 < x0 || y1 < y0)
            return;
        x0 = std::max(x0, 0);
        y0 = std::max(y0, 0);
        x1 = std::max(x1, 0);
        y1 = std::max(y1, 0);
        const PageDims d = pageDims(psm);
        const uint32_t ppr = std::max<uint32_t>(1u, (std::max<uint32_t>(bw, 1u) * 64u + d.w - 1u) / d.w);
        const uint32_t base = bp >> 5;
        const bool spill = (bp & 31u) != 0u;
        const uint32_t r0 = static_cast<uint32_t>(y0) / d.h, r1 = static_cast<uint32_t>(y1) / d.h;
        const uint32_t c0 = static_cast<uint32_t>(x0) / d.w, c1 = static_cast<uint32_t>(x1) / d.w;
        if ((r1 - r0 + 1u) * ppr >= 512u)
        {
            set.set();
            return;
        }
        for (uint32_t r = r0; r <= r1; ++r)
            for (uint32_t c = c0; c <= c1; ++c)
            {
                const uint32_t p = base + r * ppr + c;
                set.set(p & 511u);
                if (spill)
                    set.set((p + 1u) & 511u);
            }
    }

    uint32_t rgba5551To8888(uint32_t c)
    {
        return ((c & 0x1Fu) << 3) | (((c >> 5) & 0x1Fu) << 11) | (((c >> 10) & 0x1Fu) << 19) | (((c >> 15) & 1u) << 31);
    }

    uint32_t rgba8888To5551(uint32_t c)
    {
        return ((c & 0xFFu) >> 3) | ((((c >> 8) & 0xFFu) >> 3) << 5) | ((((c >> 16) & 0xFFu) >> 3) << 10) | (((c >> 31) & 1u) << 15);
    }

    using ReadFn = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
    using WriteFn = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

    ReadFn readFn(uint8_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT32: return GSMem::ReadCT32;
        case GS_PSM_CT24: return GSMem::ReadCT24;
        case GS_PSM_CT16: return GSMem::ReadCT16;
        case GS_PSM_CT16S: return GSMem::ReadCT16S;
        case GS_PSM_Z32: return GSMem::ReadZ32;
        case GS_PSM_Z24: return GSMem::ReadZ24;
        case GS_PSM_Z16: return GSMem::ReadZ16;
        case GS_PSM_Z16S: return GSMem::ReadZ16S;
        default: return GSMem::ReadNull;
        }
    }

    WriteFn writeFn(uint8_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT32: return GSMem::WriteCT32;
        case GS_PSM_CT24: return GSMem::WriteCT24;
        case GS_PSM_CT16: return GSMem::WriteCT16;
        case GS_PSM_CT16S: return GSMem::WriteCT16S;
        case GS_PSM_Z32: return GSMem::WriteZ32;
        case GS_PSM_Z24: return GSMem::WriteZ24;
        case GS_PSM_Z16: return GSMem::WriteZ16;
        case GS_PSM_Z16S: return GSMem::WriteZ16S;
        default: return GSMem::WriteNull;
        }
    }

    bool isDepthPsm(uint8_t psm) { return psm == GS_PSM_Z32 || psm == GS_PSM_Z24 || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S; }
    // Depth values are stored as z * scale in a D32_SFLOAT image (exact for 24-bit Z).
    double depthScale(uint8_t zpsm) { return zpsm == GS_PSM_Z32 ? 1.0 / 4294967296.0 : 1.0 / 16777216.0; }
    uint32_t depthMax(uint8_t zpsm)
    {
        return zpsm == GS_PSM_Z24 ? 0xFFFFFFu : (zpsm == GS_PSM_Z16 || zpsm == GS_PSM_Z16S) ? 0xFFFFu : 0xFFFFFFFFu;
    }

    bool isIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H || psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    bool validTexturePsm(uint8_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT32: case GS_PSM_CT24: case GS_PSM_CT16: case GS_PSM_CT16S:
        case GS_PSM_Z32: case GS_PSM_Z24: case GS_PSM_Z16: case GS_PSM_Z16S:
        case GS_PSM_T8: case GS_PSM_T8H: case GS_PSM_T4: case GS_PSM_T4HL: case GS_PSM_T4HH:
            return true;
        default:
            return false;
        }
    }

    struct GpuVertex
    {
        float x, y, z;
        uint8_t r, g, b, a;
        float s, t, q;
        float fog;
    };
    static_assert(sizeof(GpuVertex) == 32, "vertex layout");

    struct PushConsts
    {
        uint32_t flags;
        uint32_t atest;
        uint32_t wrap;
        uint32_t regionU;
        uint32_t regionV;
        int32_t texW;
        int32_t texH;
        uint32_t fogCol;
        int32_t kS;
        int32_t dS;
        uint32_t blend;
        uint32_t fbmsk;
        int32_t kD;
        int32_t dD;
        uint32_t texa; // ta0 | ta1 << 8 | aem << 16 (F_TEXA16 / F_TEXA24)
    };

    enum : uint32_t
    {
        F_TME = 1u,
        F_FST_TRUNC = 2u,
        F_FST_ROUND = 4u,
        F_LINEAR = 8u,
        F_TCC = 16u,
        F_FGE = 128u,
        F_IIP = 256u,
        F_BLEND_EXACT = 512u,
        F_BLEND_SCALE = 1024u,
        F_FBA = 2048u,
        F_CT16 = 4096u,
        F_ZROUND = 8192u,
        F_ALPHA_FACTOR = 16384u,
        F_SPLIT_LOW = 32768u,
        F_SPLIT_HIGH = 65536u,
        F_OUT_FM1 = 131072u,
        F_BLEND_INT = 262144u,
        F_DSTREAD = 524288u,
        F_DATE = 1048576u,
        F_BLEND_DST = 2097152u,
        F_PABE = 4194304u,
        F_AD_HALF = 8388608u,
        F_TEXA24 = 16777216u,
        F_TEXA16 = 33554432u,
        F_ZFLAT = 67108864u,
        F_BIAS_DOWN = 134217728u,
        F_BIAS_UP = 268435456u,
    };

    // Why a draw went to the CPU renderer (stats).
    enum Fallback : int
    {
        FB_PRIM,
        FB_FRAME_PSM,
        FB_ZBUF_PSM,
        FB_DATE,
        FB_PABE,
        FB_FBMSK,
        FB_BLEND_AD,
        FB_BLEND_FACTOR,
        FB_TEXTURE,
        FB_COUNT
    };
    const char *const kFallbackNames[FB_COUNT] = {"prim", "frame-psm", "zbuf-psm", "date", "pabe", "fbmsk", "blend-Ad", "blend>1", "texture"};

    struct BlendSetup
    {
        bool enable = false;
        uint8_t op = 0;  // 0 add, 1 subtract (S - D), 2 reverse subtract (D - S)
        uint8_t dst = 0; // 0 zero, 1 one, 2 src1 alpha, 3 one minus src1 alpha
        uint32_t flags = 0;
        int kS = 0, dS = 0, kD = 0, dD = 0;
        uint32_t blendWord = 0;
        bool adHalf = false; // Cs * Ad + Cd as two additive draws with the DST_ALPHA factor
    };

    struct PassSetup
    {
        bool ate = false, invert = false;
        bool rgb = true, alpha = true, depth = true;
    };

    struct DrawSetup
    {
        int fallback = -1;
        bool skip = false;          // draws nothing at all
        BlendSetup blend;
        PassSetup passes[2];
        int passCount = 0;
        uint32_t fbWriteMask = 0xFu; // RGBA bits the frame accepts (FBMSK, CT24)
        bool needDepth = false;
        uint8_t ztest = 1;
        bool dstRead = false; // the shader reads a copy of the target (DATE, FBMSK bits, blends)
    };

    DrawSetup analyseState(const GSDrawState &st)
    {
        DrawSetup d;
        const GSContext &ctx = st.context;
        switch (st.prim.type)
        {
        case GS_PRIM_TRIANGLE:
        case GS_PRIM_TRISTRIP:
        case GS_PRIM_TRIFAN:
        case GS_PRIM_SPRITE:
        case GS_PRIM_POINT:
        case GS_PRIM_LINE:
        case GS_PRIM_LINESTRIP:
            break;
        default:
            d.fallback = FB_PRIM;
            return d;
        }
        // Lines and points are untextured on the GS path the CPU renderer implements.
        const bool usesTexture = st.prim.tme && st.prim.type != GS_PRIM_POINT && st.prim.type != GS_PRIM_LINE && st.prim.type != GS_PRIM_LINESTRIP;
        const uint8_t fpsm = ctx.frame.psm;
        if (fpsm != GS_PSM_CT32 && fpsm != GS_PSM_CT24 && fpsm != GS_PSM_CT16)
        {
            d.fallback = FB_FRAME_PSM;
            return d;
        }
        bool dst = false;
        uint32_t extraFlags = 0;
        // DATE (RGB24 has no destination alpha: always passes).
        if (((ctx.test >> 14) & 1u) && fpsm != GS_PSM_CT24)
        {
            dst = true;
            extraFlags |= F_DATE;
        }
        if (st.pabe && st.prim.abe)
        {
            dst = true;
            extraFlags |= F_PABE;
        }
        // FBMSK: whole bytes become the channel write mask; other bits need the destination.
        uint32_t mask = 0xFu;
        {
            const uint32_t m = ctx.frame.fbmsk;
            for (int c = 0; c < 4; ++c)
            {
                const uint32_t b = (m >> (c * 8)) & 0xFFu;
                if (b == 0xFFu)
                    mask &= ~(1u << c);
                else if (b != 0u)
                    dst = true;
            }
        }
        if (fpsm == GS_PSM_CT24)
            mask &= 0x7u;
        d.fbWriteMask = mask;

        if (usesTexture)
        {
            const uint8_t tpsm = ctx.tex0.psm;
            if (!validTexturePsm(tpsm) || st.textureWidth > 1024u || st.textureHeight > 1024u)
            {
                d.fallback = FB_TEXTURE;
                return d;
            }
        }

        // Blend.
        BlendSetup &b = d.blend;
        if (st.prim.abe)
        {
            const uint64_t ar = ctx.alpha;
            const uint32_t asel = ar & 3u, bsel = (ar >> 2) & 3u, csel = (ar >> 4) & 3u, dsel = (ar >> 6) & 3u;
            const int kS = int(asel == 0) - int(bsel == 0);
            const int kD = int(asel == 1) - int(bsel == 1);
            const int dS = int(dsel == 0);
            const int dD = int(dsel == 1);
            b.kS = kS;
            b.dS = dS;
            b.kD = kD;
            b.dD = dD;
            b.blendWord = static_cast<uint32_t>((ar >> 32) & 0xFFu) | ((csel == 2u || csel == 3u) ? 0x100u : 0u) | (csel == 1u ? 0x200u : 0u) |
                          ((st.colclamp & 1u) ? 0x800u : 0u);
            if (dst || (csel == 1u && (kS != 0 || kD != 0) && !(kS == 1 && kD == 0 && dS == 0 && dD == 1)))
            {
                dst = true; // blended in the shader
            }
            else if (csel == 1u && kS == 1 && kD == 0 && dS == 0 && dD == 1)
            {
                b.enable = true;
                b.adHalf = true;
            }
            else if (kD == 0 && dD == 0)
                b.flags = F_BLEND_EXACT;
            else if (kD == 0 && dD == 1)
            {
                // Cd + source term: the source term is computed exactly in the shader.
                b.enable = true;
                b.dst = 1;
                b.op = (kS == -1 && dS == 0) ? 2 : 0;
                b.flags = F_BLEND_INT;
            }
            else if (kD == 1 && dD == 0)
            {
                b.enable = true;
                b.dst = 2;
                b.op = (kS == -1 && dS == 0) ? 2 : 0;
                b.flags = F_BLEND_SCALE;
            }
            else if (kD == -1 && dD == 0)
            {
                if (kS == 0 && dS == 0)
                    b.flags = F_BLEND_EXACT; // -Cd * f clamps to 0
                else
                {
                    b.enable = true;
                    b.dst = 2;
                    b.op = 1;
                    b.flags = F_BLEND_SCALE;
                }
            }
            else if (kD == -1 && dD == 1)
            {
                b.enable = true;
                b.dst = 3;
                b.op = 0;
                b.flags = F_BLEND_SCALE;
            }
            else
                dst = true; // Cd * (1 + C): blended in the shader
        }
        if (dst)
        {
            // The shader does blend, FBMSK and FBA itself: plain writes of every channel it may change.
            d.dstRead = true;
            b.enable = false;
            b.adHalf = false;
            b.flags = F_DSTREAD | extraFlags | (st.prim.abe ? F_BLEND_DST : 0u);
            d.fbWriteMask = fpsm == GS_PSM_CT24 ? 0x7u : 0xFu;
            b.blendWord |= ((ctx.test >> 15) & 1u) ? 0x400u : 0u;
            mask = d.fbWriteMask;
        }

        // Alpha test / AFAIL -> up to two passes.
        const uint64_t test = ctx.test;
        const bool ate = (test & 1u) != 0u;
        const uint32_t atst = (test >> 1) & 7u;
        const uint32_t afail = (test >> 12) & 3u;
        PassSetup normal{};
        PassSetup failed{};
        bool failDraws = true;
        switch (afail)
        {
        case 0: failDraws = false; break;
        case 1: failed.rgb = true; failed.alpha = true; failed.depth = false; break;
        case 2: failed.rgb = false; failed.alpha = false; failed.depth = true; break;
        default:
            failed.rgb = true;
            failed.alpha = fpsm != GS_PSM_CT32;
            failed.depth = false;
            break;
        }
        if (!ate || atst == 1u)
            d.passes[d.passCount++] = normal;
        else if (atst == 0u)
        {
            if (failDraws)
                d.passes[d.passCount++] = failed;
        }
        else
        {
            normal.ate = true;
            d.passes[d.passCount++] = normal;
            if (failDraws)
            {
                failed.ate = true;
                failed.invert = true;
                d.passes[d.passCount++] = failed;
            }
        }

        d.ztest = static_cast<uint8_t>((test >> 17) & 3u);
        if (d.ztest == 0u)
        {
            d.skip = true;
            return d;
        }
        const bool zmask = ctx.zbuf.zmask;
        bool anyWrite = false;
        bool writesZ = false;
        for (int i = 0; i < d.passCount; ++i)
        {
            PassSetup &p = d.passes[i];
            p.depth = p.depth && !zmask;
            const uint32_t cm = ((p.rgb ? 7u : 0u) | (p.alpha ? 8u : 0u)) & mask;
            anyWrite = anyWrite || cm != 0u || p.depth;
            writesZ = writesZ || p.depth;
        }
        if (!anyWrite)
        {
            d.skip = true;
            return d;
        }
        d.needDepth = d.ztest >= 2u || writesZ;
        if (d.needDepth)
        {
            const uint8_t zpsm = ctx.zbuf.psm;
            if (zpsm != GS_PSM_Z24 && zpsm != GS_PSM_Z32 && zpsm != GS_PSM_Z16)
            {
                d.fallback = FB_ZBUF_PSM;
                return d;
            }
        }
        return d;
    }

    bool primitiveOutsideScissor(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &state = batch.state;
        const GSContext &ctx = state.context;
        const uint32_t count = (state.prim.type == GS_PRIM_SPRITE || state.prim.type == GS_PRIM_LINE || state.prim.type == GS_PRIM_LINESTRIP) ? 2u
                               : (state.prim.type == GS_PRIM_POINT ? 1u : 3u);
        if (batch.vertexCount < count)
            return false;
        const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        float xMin = batch.vertices[0].x, xMax = xMin, yMin = batch.vertices[0].y, yMax = yMin;
        for (uint32_t i = 1; i < count; ++i)
        {
            xMin = std::min(xMin, batch.vertices[i].x);
            xMax = std::max(xMax, batch.vertices[i].x);
            yMin = std::min(yMin, batch.vertices[i].y);
            yMax = std::max(yMax, batch.vertices[i].y);
        }
        if (!(xMin == xMin && xMax == xMax && yMin == yMin && yMax == yMax))
            return false;
        return xMax - ofx + 1.0f < static_cast<float>(ctx.scissor.x0) || xMin - ofx - 1.0f > static_cast<float>(ctx.scissor.x1) ||
               yMax - ofy + 1.0f < static_cast<float>(ctx.scissor.y0) || yMin - ofy - 1.0f > static_cast<float>(ctx.scissor.y1);
    }

    uint64_t hashWords(const uint32_t *p, size_t n, uint64_t h = 1469598103934665603ull)
    {
        for (size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ull;
            h ^= h >> 29;
        }
        return h;
    }

    uint64_t nowNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }
}

class GsVulkanBackend final : public GSRasterBackend
{
public:
    GsVulkanBackend() = default;
    ~GsVulkanBackend() override;
    bool Create(const struct HostVulkanShared *shared);

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;
    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;
    void Flush() override;
    void TextureFlush() override {}
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;
    void QueuePresentSnapshot(const GSPresentationRequest &request) override;
    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;
    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;
    bool PresentsOnGpu() const override;

private:
    struct GpuImage
    {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        uint32_t width = 0, height = 0;
    };

    struct Target
    {
        uint32_t fbp = 0, fbw = 1;
        uint8_t psm = 0;
        bool depth = false;
        GpuImage img;
        uint64_t stale = 0;    // row groups (page height) to upload from GS memory before use
        bool dirty = false;    // [dx0,dx1] x [dy0,dy1] is newer than GS memory
        int dx0 = 0, dy0 = 0, dx1 = -1, dy1 = -1;
        PageSet dirtyPages;
        uint64_t lastUse = 0;
        uint64_t version = 1; // bumped whenever the image content changes
        PageSet allPages;     // every page the target's rows cover
        int maxRow = -1;      // lowest row any draw has reached
    };

    struct TexKey
    {
        uint32_t tbp0, tbw, psm, w, h, texa;
        uint64_t palette;
        bool operator==(const TexKey &o) const
        {
            return tbp0 == o.tbp0 && tbw == o.tbw && psm == o.psm && w == o.w && h == o.h && texa == o.texa && palette == o.palette;
        }
    };
    struct TexKeyHash
    {
        size_t operator()(const TexKey &k) const
        {
            const uint32_t w[6] = {k.tbp0, k.tbw, k.psm, k.w, k.h, k.texa};
            return static_cast<size_t>(hashWords(w, 6, k.palette));
        }
    };
    struct Texture
    {
        GpuImage img;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::vector<uint16_t> pages;
        uint64_t serial = 0;  // GS memory serial the texels were decoded at
        uint64_t rawHash = 0; // hash of its pages' raw GS memory at decode time
        uint64_t mirrorGen = 0; // m_mirrorGenCounter at decode (GPU decode)
        uint64_t lastUse = 0; // m_submitSerial of the last command buffer using it
        uint8_t maxAlpha = 0; // largest texel alpha (an upper bound until the GPU reports it)
        uint32_t alphaSlot = ~0u; // m_alphaRes slot the GPU decode reports into
        uint64_t alphaId = 0, alphaSerial = 0;
    };

    struct Batch
    {
        bool active = false;
        GSDrawState state{};
        DrawSetup setup;
        std::vector<GpuVertex> verts;
        float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
        uint32_t maxVA = 0; // largest vertex alpha
    };

    // ---- Vulkan plumbing ----
    bool createDeviceObjects();
    uint32_t findMemoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback = 0) const;
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer &buf, VkDeviceMemory &mem, void **mapped, bool *coherent = nullptr);
    bool createImage(GpuImage &img, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect);
    void destroyImage(GpuImage &img);
    void beginCmd();
    void endRendering();
    void barrier();
    void submitAndWait();
    uint32_t ringAlloc(uint32_t size, uint32_t align = 16u);
    VkPipeline getPipeline(uint32_t key);

    // ---- coherency ----
    Target &getTarget(uint32_t fbp, uint32_t fbw, uint8_t psm);
    uint64_t groupsFor(const Target &t, const PageSet &pages) const;
    PageSet groupPages(const Target &t, uint64_t groups) const;
    void ensureVramCurrent(const PageSet &pages);
    void markCpuWrite(const PageSet &pages);
    void bumpPages(const PageSet &pages);
    void download(Target &t);
    void downloadAll(bool includeDepth);
    void uploadStale(Target &t, int y0, int y1);
    void prepareDrawTarget(Target &t, const PageSet &drawPages, int y0, int y1);
    void dropAll();
    uint64_t pagesHash(const std::vector<uint16_t> &pages) const;
    static PageSet displayPages(const GSPresentationRequest &request);

    // ---- drawing ----
    void appendPrimitive(const GSPrimitiveBatch &batch);
    void flushBatch();
    Texture *getTexture(const GSDrawState &st, uint32_t &decW, uint32_t &decH, uint32_t &texFlags);
    void gpuDecode(const GSDrawState &st, Texture &tex, const PageSet &pages, const uint32_t *palette);
    void resolveMaxAlpha(Texture &t);
    void syncMirrorPages(const PageSet &pages);
    void overlayTarget(Target &t);
    uint8_t estimateMaxAlpha(const GSDrawState &st, const uint32_t *palette, bool overlay, const std::vector<uint16_t> &pages) const;
    static int decodeKind(uint8_t psm);
    VkDescriptorSet computeSet();
    PageSet dirtyTargetPages() const;
    Texture *getAliasTexture(const GSDrawState &st, uint32_t w, uint32_t h, const PageSet &pages, uint32_t &texFlags);
    void cpuDraw(const GSPrimitiveBatch &batch, int reason);
    void printStats();

    // Vulkan objects
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_phys = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;
    VolkInstanceTable m_it{};
    VolkDeviceTable m_dt{};
    VkPhysicalDeviceMemoryProperties m_memProps{};
    bool m_depthClamp = false;
    bool m_dualSrc = false;
    VkCommandPool m_cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
    // Two command buffers: one recording while the other may still execute (flips submit without
    // waiting). m_cmd / m_fence / m_compPool are the current slot's.
    struct Slot
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkDescriptorPool compPool = VK_NULL_HANDLE;
        bool pending = false;
        uint64_t serial = 0;
        std::vector<GpuImage> defImages;
        std::vector<VkDescriptorSet> defSets;
        // Display pages copied out of the mirror at a flip, published once the slot completes.
        VkBuffer presentBuf = VK_NULL_HANDLE;
        VkDeviceMemory presentMem = VK_NULL_HANDLE;
        uint8_t *presentPtr = nullptr;
        bool presentCoherent = true;
        bool hasPresent = false;
        std::vector<uint16_t> presentPages;
        GSPresentationRequest presentReq{};
        // GPU presentation: the display pass of this submission writes m_presentImg[gpuImage]
        // (published at submit, ready when m_readySem reaches gpuValue).
        int gpuImage = -1;
        uint64_t gpuValue = 0;
        uint32_t gpuW = 0, gpuH = 0;
        // PS2_GS_VK_CHECKPRESENT: the display image read back, compared with the CPU picture.
        VkBuffer checkBuf = VK_NULL_HANDLE;
        VkDeviceMemory checkMem = VK_NULL_HANDLE;
        uint8_t *checkPtr = nullptr;
        bool checkCoherent = true;
        bool hasCheck = false;
    };
    Slot m_slots[2];
    uint32_t m_cur = 0;
    uint32_t m_poolUsed = 0;
    uint64_t m_completedSerial = 0; // every command buffer with a lower serial has completed
    void submitAsync();
    void queueSubmit();
    void completeSlot(uint32_t i);
    void reserve(uint32_t ringBytes, uint32_t sets);
    bool m_cmdOpen = false;
    uint64_t m_submitSerial = 1;
    VkBuffer m_ring = VK_NULL_HANDLE;
    VkDeviceMemory m_ringMem = VK_NULL_HANDLE;
    uint8_t *m_ringPtr = nullptr;
    uint32_t m_ringOffset = 0;
    VkBuffer m_readback = VK_NULL_HANDLE;
    VkDeviceMemory m_readbackMem = VK_NULL_HANDLE;
    uint8_t *m_readbackPtr = nullptr;
    bool m_readbackCoherent = true;
    VkDeviceSize m_readbackSize = 0;
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    // GPU mirror of GS memory (texture decode on the GPU) and the swizzle tables.
    VkBuffer m_vramBuf = VK_NULL_HANDLE, m_lutBuf = VK_NULL_HANDLE;
    VkDeviceMemory m_vramMem = VK_NULL_HANDLE, m_lutMem = VK_NULL_HANDLE;
    std::array<uint64_t, 512> m_mirrorSerial{}; // m_pageSerial the mirror page holds (~0 = stale)
    std::array<uint64_t, 512> m_mirrorGen{};    // bumped whenever a mirror page changes
    uint64_t m_mirrorGenCounter = 1;
    PageSet m_mirrorNewer;                      // mirror pages newer than GS memory (render-target data)
    VkBuffer m_scratch = VK_NULL_HANDLE;        // linear target rows read out of the mirror
    VkDeviceMemory m_scratchMem = VK_NULL_HANDLE;
    VkPipeline m_unswizzlePipe = VK_NULL_HANDLE;
    bool gpuMem() const { return !m_cpuDecode; }
    void ensureMirrorCurrent(const PageSet &pages);
    void readbackMirror(const PageSet &pages);
    bool ensureReadback(VkDeviceSize bytes);
    void uploadFromMirror(Target &t, uint32_t y0, uint32_t rows);
    uint32_t m_lutOffset[8]{};                  // first entry of each table (C32, C16, C16S, P8, P4, Z32, Z16, Z16S)
    VkDescriptorSetLayout m_compLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_compPipeLayout = VK_NULL_HANDLE;
    VkPipeline m_decodePipe = VK_NULL_HANDLE, m_overlayPipe = VK_NULL_HANDLE;
    VkDescriptorPool m_compPool = VK_NULL_HANDLE;
    VkDeviceSize m_ssboAlign = 256;
    bool m_cpuDecode = false;
    // Largest texel alpha of each GPU decode, written by the decode shader (host-visible).
    static constexpr uint32_t kAlphaSlots = 4096u;
    VkBuffer m_alphaRes = VK_NULL_HANDLE;
    VkDeviceMemory m_alphaResMem = VK_NULL_HANDLE;
    uint32_t *m_alphaResPtr = nullptr;
    std::vector<uint64_t> m_alphaOwner;
    uint32_t m_alphaNext = 0;
    uint64_t m_alphaIds = 0;
    bool createComputeObjects();
    GpuImage m_dstImg;                   // copy of the target for draws that read it (set 1)
    VkDescriptorSet m_dstSet = VK_NULL_HANDLE;
    bool ensureDstImage(uint32_t width);
    VkPipelineLayout m_pipeLayout = VK_NULL_HANDLE;
    VkShaderModule m_vs = VK_NULL_HANDLE, m_fs = VK_NULL_HANDLE;
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    std::unordered_map<uint32_t, VkPipeline> m_pipelines;
    GpuImage m_dummyTex;
    VkDescriptorSet m_dummySet = VK_NULL_HANDLE;

    // Rendering state
    bool m_rendering = false;
    Target *m_curColor = nullptr;
    Target *m_curDepth = nullptr;
    VkPipeline m_curPipeline = VK_NULL_HANDLE;

    // GS state
    mutable std::recursive_mutex m_mutex;
    GSCpuBackend m_cpu;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::vector<Target *> m_targets;
    std::unordered_map<TexKey, Texture, TexKeyHash> m_textures;
    std::unordered_map<Target *, Texture> m_alias; // render targets sampled directly as textures
    std::array<uint64_t, 512> m_pageSerial{};
    uint64_t m_serial = 1;
    Batch m_batch;
    GSDrawState m_lastAnalysed{};
    DrawSetup m_lastSetup;
    bool m_haveLastAnalysed = false;
    PageSet m_uploadPages;
    std::vector<uint32_t> m_decodeBuf;
    struct PaletteEntry
    {
        std::array<uint32_t, 256> colors{};
        uint64_t hash = 0;
    };
    std::unordered_map<uint64_t, PaletteEntry> m_palCache; // decoded palettes for the current CLUT
    uint64_t m_palVersion = ~0ull;

    // GPU presentation: at each flip the displayed picture is built from the mirror into one of
    // these images, which the host blits to the swapchain (no trip through the CPU).
    struct DispSrc
    {
        uint32_t valid = 0, bp = 0, bw = 0, psm = 0, ox = 0, oy = 0, w = 0, h = 0;
    };
    struct DispParams // matches Params in shaders/gs_display.comp (std430)
    {
        DispSrc src[6];
        uint32_t circuits = 0, fallback = 0, outW = 0, outH = 0;
        uint32_t mmod = 0, amod = 0, slbg = 0, alp = 0;
        uint32_t bgcolor = 0, bob = 0, oddField = 0;
        uint32_t tables[3]{};
    };
    static constexpr uint32_t kPresentImages = 4u; // published + up to two being shown + one to draw
    static constexpr uint32_t kPresentW = 640u, kPresentH = 512u;
    bool buildDisplayParams(const GSPresentationRequest &request, DispParams &dp) const;
    int pickPresentImage();
    void recordDisplay(const DispParams &dp, int image);
    void checkPresent(Slot &sl, const std::vector<uint8_t> &snap);
    static bool provideFrame(void *user, struct HostGpuFrame &out, uint64_t releaseValue);
    bool m_gpuPresent = false;   // build the picture on the GPU at flips
    bool m_hostPresent = false;  // ... and the host shows it
    bool m_checkPresent = false;
    bool m_sharedDevice = false;
    std::mutex *m_queueMutex = nullptr; // the host's queue lock when the device is shared
    VkSemaphore m_releaseSem = VK_NULL_HANDLE; // host's: images it has finished reading
    VkSemaphore m_readySem = VK_NULL_HANDLE;   // ours: display passes done
    uint64_t m_readyValue = 0;
    GpuImage m_presentImg[kPresentImages];
    VkBuffer m_dispCount = VK_NULL_HANDLE;
    VkDeviceMemory m_dispCountMem = VK_NULL_HANDLE;
    VkPipeline m_displayPipe = VK_NULL_HANDLE;
    mutable std::mutex m_gpuFrameMutex; // the published frame and m_presentLastUse
    uint64_t m_presentLastUse[kPresentImages]{};
    uint32_t m_presentNext = 0;
    int m_pubImage = -1;
    uint32_t m_pubW = 0, m_pubH = 0;
    uint64_t m_pubValue = 0;
    bool m_pubGpu = false; // the last flip went through the GPU display pass
    uint64_t m_pubNs = 0;
    mutable bool m_hostUseGpu = true; // PresentsOnGpu()'s last answer: the host frame follows it
    uint64_t m_statGpuFlips = 0, m_statCheckBad = 0, m_statChecks = 0;

    // Presentation
    std::mutex m_presentMutex;
    std::vector<uint8_t> m_snapLatest;
    GSPresentationRequest m_snapRequest{};
    bool m_haveSnap = false;
    std::atomic<uint64_t> m_lastFlipNs{0};
    std::vector<uint8_t> m_presentVram;
    GSCpuBackend m_presenter;

    // Stats
    bool m_stats = false;
    uint64_t m_statBatches = 0, m_statDraws = 0, m_statPrims = 0, m_statDownloads = 0, m_statDownloadPx = 0;
    uint64_t m_statUploads = 0, m_statUploadRows = 0, m_statTexUploads = 0, m_statTexHits = 0, m_statSubmits = 0;
    uint64_t m_statFallback[FB_COUNT]{};
    uint64_t m_statFlips = 0, m_ivFlips = 0;
    uint64_t m_statDstCopies = 0, m_statAliasCopies = 0, m_statTexRehash = 0, m_statOverlays = 0, m_statMirrorPages = 0;
    // Interval timings (ns) and counts, printed per frame with the stats.
    struct Interval
    {
        uint64_t submitNs = 0, waitNs = 0, downloadNs = 0, decodeNs = 0, uploadNs = 0, flushNs = 0, xferNs = 0, clutNs = 0, flipNs = 0;
        uint64_t downloads = 0, decodes = 0, uploads = 0, batches = 0, prims = 0, waits = 0;
    } m_iv;
    struct ScopeTimer
    {
        uint64_t &acc;
        uint64_t t0;
        explicit ScopeTimer(uint64_t &a) : acc(a), t0(nowNs()) {}
        ~ScopeTimer() { acc += nowNs() - t0; }
    };
    bool m_noAlias = false;
    bool m_noBias = false;
    bool m_noBigDst = false;
    bool m_syncPresent = false;
    const char *m_why = "?";
    std::unordered_map<std::string, uint64_t> m_statWhy;
    uint64_t m_statBigFactor[256]{};
};

// ------------------------------------------------------------------------------------------
// Vulkan setup
// ------------------------------------------------------------------------------------------

GsVulkanBackend::~GsVulkanBackend()
{
#if defined(PS2X_HOST_SDL3)
    if (m_hostPresent)
        HostVulkanSetFrameProvider(nullptr, nullptr);
#endif
    if (!m_device)
    {
        if (m_instance && !m_sharedDevice)
            m_it.vkDestroyInstance ? m_it.vkDestroyInstance(m_instance, nullptr) : void();
        return;
    }
    if (m_stats)
        printStats();
    if (m_queueMutex)
    {
        std::lock_guard<std::mutex> qlock(*m_queueMutex);
        m_dt.vkDeviceWaitIdle(m_device);
    }
    else
        m_dt.vkDeviceWaitIdle(m_device);
    for (GpuImage &img : m_presentImg)
        destroyImage(img);
    if (m_displayPipe) m_dt.vkDestroyPipeline(m_device, m_displayPipe, nullptr);
    if (m_dispCount) m_dt.vkDestroyBuffer(m_device, m_dispCount, nullptr);
    if (m_dispCountMem) m_dt.vkFreeMemory(m_device, m_dispCountMem, nullptr);
    if (m_readySem) m_dt.vkDestroySemaphore(m_device, m_readySem, nullptr);
    for (Target *t : m_targets)
    {
        destroyImage(t->img);
        delete t;
    }
    m_targets.clear();
    for (auto &kv : m_textures)
        destroyImage(kv.second.img);
    m_textures.clear();
    for (auto &kv : m_alias)
        destroyImage(kv.second.img);
    m_alias.clear();
    destroyImage(m_dummyTex);
    destroyImage(m_dstImg);
    for (auto &kv : m_pipelines)
        m_dt.vkDestroyPipeline(m_device, kv.second, nullptr);
    if (m_descPool) m_dt.vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    if (m_decodePipe) m_dt.vkDestroyPipeline(m_device, m_decodePipe, nullptr);
    if (m_overlayPipe) m_dt.vkDestroyPipeline(m_device, m_overlayPipe, nullptr);
    if (m_unswizzlePipe) m_dt.vkDestroyPipeline(m_device, m_unswizzlePipe, nullptr);
    if (m_scratch) m_dt.vkDestroyBuffer(m_device, m_scratch, nullptr);
    if (m_scratchMem) m_dt.vkFreeMemory(m_device, m_scratchMem, nullptr);
    if (m_compPipeLayout) m_dt.vkDestroyPipelineLayout(m_device, m_compPipeLayout, nullptr);
    if (m_compLayout) m_dt.vkDestroyDescriptorSetLayout(m_device, m_compLayout, nullptr);
    if (m_vramBuf) m_dt.vkDestroyBuffer(m_device, m_vramBuf, nullptr);
    if (m_vramMem) m_dt.vkFreeMemory(m_device, m_vramMem, nullptr);
    if (m_lutBuf) m_dt.vkDestroyBuffer(m_device, m_lutBuf, nullptr);
    if (m_lutMem) m_dt.vkFreeMemory(m_device, m_lutMem, nullptr);
    if (m_alphaRes) m_dt.vkDestroyBuffer(m_device, m_alphaRes, nullptr);
    if (m_alphaResMem) m_dt.vkFreeMemory(m_device, m_alphaResMem, nullptr);
    if (m_vs) m_dt.vkDestroyShaderModule(m_device, m_vs, nullptr);
    if (m_fs) m_dt.vkDestroyShaderModule(m_device, m_fs, nullptr);
    if (m_pipeLayout) m_dt.vkDestroyPipelineLayout(m_device, m_pipeLayout, nullptr);
    if (m_setLayout) m_dt.vkDestroyDescriptorSetLayout(m_device, m_setLayout, nullptr);
    if (m_sampler) m_dt.vkDestroySampler(m_device, m_sampler, nullptr);
    if (m_ring) m_dt.vkDestroyBuffer(m_device, m_ring, nullptr);
    if (m_ringMem) m_dt.vkFreeMemory(m_device, m_ringMem, nullptr);
    if (m_readback) m_dt.vkDestroyBuffer(m_device, m_readback, nullptr);
    if (m_readbackMem) m_dt.vkFreeMemory(m_device, m_readbackMem, nullptr);
    for (Slot &sl : m_slots)
    {
        if (sl.fence) m_dt.vkDestroyFence(m_device, sl.fence, nullptr);
        if (sl.compPool) m_dt.vkDestroyDescriptorPool(m_device, sl.compPool, nullptr);
        if (sl.presentBuf) m_dt.vkDestroyBuffer(m_device, sl.presentBuf, nullptr);
        if (sl.presentMem) m_dt.vkFreeMemory(m_device, sl.presentMem, nullptr);
        if (sl.checkBuf) m_dt.vkDestroyBuffer(m_device, sl.checkBuf, nullptr);
        if (sl.checkMem) m_dt.vkFreeMemory(m_device, sl.checkMem, nullptr);
        for (GpuImage &img : sl.defImages)
            destroyImage(img);
    }
    if (m_cmdPool) m_dt.vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
    if (m_sharedDevice)
    {
#if defined(PS2X_HOST_SDL3)
        HostVulkanRelease(); // the host's device: it goes when the host is done with it too
#endif
        return;
    }
    m_dt.vkDestroyDevice(m_device, nullptr);
    m_it.vkDestroyInstance(m_instance, nullptr);
}

bool GsVulkanBackend::Create(const HostVulkanShared *shared)
{
    m_stats = envFlag("PS2_GS_VK_STATS");
    m_checkPresent = envFlag("PS2_GS_VK_CHECKPRESENT");
    m_noAlias = envFlag("PS2_GS_VK_NOALIAS");
    m_noBias = envFlag("PS2_GS_VK_NOBIAS");
    m_cpuDecode = envFlag("PS2_GS_VK_CPUDECODE");
    m_noBigDst = envFlag("PS2_GS_VK_SPLITBLEND");
    m_syncPresent = envFlag("PS2_GS_VK_SYNCPRESENT");
    m_cpu.SetSynchronous();
    m_presenter.SetSynchronous();
#if defined(PS2X_HOST_SDL3)
    if (shared && !envFlag("PS2_GS_VK_OWNDEVICE"))
    {
        // The host's device: what is drawn here can be shown without leaving the GPU.
        m_sharedDevice = true;
        HostVulkanRetain();
        m_instance = shared->instance;
        m_phys = shared->physical;
        m_device = shared->device;
        m_queue = shared->queue;
        m_queueFamily = shared->queueFamily;
        m_queueMutex = shared->queueMutex;
        m_releaseSem = shared->releaseSemaphore;
        m_depthClamp = shared->depthClamp;
        m_dualSrc = shared->dualSrcBlend && envFlag("PS2_GS_VK_DUALSRC");
        volkLoadInstanceTable(&m_it, m_instance);
        volkLoadDeviceTable(&m_dt, m_device);
        VkPhysicalDeviceProperties props{};
        m_it.vkGetPhysicalDeviceProperties(m_phys, &props);
        std::fprintf(stderr, "[gs:vk] using %s (shared with the window)%s\n", props.deviceName, m_dualSrc ? " (dual-source blending)" : "");
        m_it.vkGetPhysicalDeviceMemoryProperties(m_phys, &m_memProps);
        m_gpuPresent = !envFlag("PS2_GS_VK_CPUPRESENT");
        if (!createDeviceObjects())
            return false;
        if (m_gpuPresent)
        {
            m_hostPresent = true;
            HostVulkanSetFrameProvider(&GsVulkanBackend::provideFrame, this);
        }
        return true;
    }
#else
    (void)shared;
#endif
    if (volkInitialize() != VK_SUCCESS)
    {
        std::fprintf(stderr, "[gs:vk] no Vulkan loader\n");
        return false;
    }
    uint32_t apiVersion = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion)
        vkEnumerateInstanceVersion(&apiVersion);
    if (apiVersion < VK_API_VERSION_1_3)
    {
        std::fprintf(stderr, "[gs:vk] Vulkan 1.3 instance not available\n");
        return false;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ps2recomp-gs";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    const char *validation = "VK_LAYER_KHRONOS_validation";
    if (envFlag("PS2_VK_VALIDATION"))
    {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &validation;
    }
    if (vkCreateInstance(&ici, nullptr, &m_instance) != VK_SUCCESS)
    {
        ici.enabledLayerCount = 0;
        if (vkCreateInstance(&ici, nullptr, &m_instance) != VK_SUCCESS)
        {
            std::fprintf(stderr, "[gs:vk] vkCreateInstance failed\n");
            return false;
        }
    }
    volkLoadInstanceTable(&m_it, m_instance);
    if (!vkGetDeviceProcAddr)
        vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(vkGetInstanceProcAddr(m_instance, "vkGetDeviceProcAddr"));

    uint32_t count = 0;
    m_it.vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devs(count);
    m_it.vkEnumeratePhysicalDevices(m_instance, &count, devs.data());
    int forced = -1;
    if (const char *v = std::getenv("PS2_VK_GPU"))
        forced = std::atoi(v);
    int bestScore = -1;
    for (uint32_t i = 0; i < count; ++i)
    {
        VkPhysicalDeviceProperties props{};
        m_it.vkGetPhysicalDeviceProperties(devs[i], &props);
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f13;
        m_it.vkGetPhysicalDeviceFeatures2(devs[i], &f2);
        uint32_t qn = 0;
        m_it.vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        m_it.vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qf.data());
        int family = -1;
        for (uint32_t q = 0; q < qn; ++q)
            if (qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                family = static_cast<int>(q);
                break;
            }
        const bool ok = props.apiVersion >= VK_API_VERSION_1_3 && f13.dynamicRendering && family >= 0;
        std::fprintf(stderr, "[gs:vk] device %u: %s (api %u.%u)%s\n", i, props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
                     VK_API_VERSION_MINOR(props.apiVersion), ok ? "" : " - unsuitable");
        if (!ok)
            continue;
        int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if (forced == static_cast<int>(i))
            score = 100;
        if (score > bestScore)
        {
            bestScore = score;
            m_phys = devs[i];
            m_queueFamily = static_cast<uint32_t>(family);
            m_depthClamp = f2.features.depthClamp != VK_FALSE;
            // Dual-source blending saves the separate alpha pass of blended draws; opt-in until it
            // has been checked against the CPU renderer on real hardware (PS2_GS_VK_DUALSRC=1).
            m_dualSrc = f2.features.dualSrcBlend != VK_FALSE && envFlag("PS2_GS_VK_DUALSRC");
        }
    }
    if (!m_phys)
    {
        std::fprintf(stderr, "[gs:vk] no suitable device (needs Vulkan 1.3 with dynamic rendering)\n");
        return false;
    }
    {
        VkPhysicalDeviceProperties props{};
        m_it.vkGetPhysicalDeviceProperties(m_phys, &props);
        std::fprintf(stderr, "[gs:vk] using %s%s\n", props.deviceName, m_dualSrc ? " (dual-source blending)" : "");
    }
    m_it.vkGetPhysicalDeviceMemoryProperties(m_phys, &m_memProps);

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = m_queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f13;
    f2.features.dualSrcBlend = m_dualSrc ? VK_TRUE : VK_FALSE;
    f2.features.depthClamp = m_depthClamp ? VK_TRUE : VK_FALSE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (m_it.vkCreateDevice(m_phys, &dci, nullptr, &m_device) != VK_SUCCESS)
    {
        std::fprintf(stderr, "[gs:vk] vkCreateDevice failed\n");
        return false;
    }
    volkLoadDeviceTable(&m_dt, m_device);
    m_dt.vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);
    // Own device (no window to show on): the display pass only runs to be checked.
    m_gpuPresent = m_checkPresent || envFlag("PS2_GS_VK_GPUPRESENT");
    return createDeviceObjects();
}

uint32_t GsVulkanBackend::findMemoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback) const
{
    for (int pass = 0; pass < 2; ++pass)
    {
        const VkMemoryPropertyFlags f = pass == 0 ? want : fallback;
        if (pass == 1 && fallback == 0)
            break;
        for (uint32_t i = 0; i < m_memProps.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (m_memProps.memoryTypes[i].propertyFlags & f) == f)
                return i;
    }
    return UINT32_MAX;
}

bool GsVulkanBackend::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer &buf,
                                   VkDeviceMemory &mem, void **mapped, bool *coherent)
{
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    if (m_dt.vkCreateBuffer(m_device, &bci, nullptr, &buf) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req{};
    m_dt.vkGetBufferMemoryRequirements(m_device, buf, &req);
    uint32_t type = findMemoryType(req.memoryTypeBits, props, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX)
        return false;
    if (coherent)
        *coherent = (m_memProps.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (m_dt.vkAllocateMemory(m_device, &mai, nullptr, &mem) != VK_SUCCESS)
        return false;
    m_dt.vkBindBufferMemory(m_device, buf, mem, 0);
    if (mapped && m_dt.vkMapMemory(m_device, mem, 0, VK_WHOLE_SIZE, 0, mapped) != VK_SUCCESS)
        return false;
    return true;
}

bool GsVulkanBackend::createImage(GpuImage &img, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
{
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (m_dt.vkCreateImage(m_device, &ici, nullptr, &img.image) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req{};
    m_dt.vkGetImageMemoryRequirements(m_device, img.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (mai.memoryTypeIndex == UINT32_MAX)
        mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, 0, 0);
    if (m_dt.vkAllocateMemory(m_device, &mai, nullptr, &img.memory) != VK_SUCCESS)
        return false;
    m_dt.vkBindImageMemory(m_device, img.image, img.memory, 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = img.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    if (m_dt.vkCreateImageView(m_device, &vci, nullptr, &img.view) != VK_SUCCESS)
        return false;
    img.width = w;
    img.height = h;
    // Every image lives in GENERAL layout.
    beginCmd();
    endRendering();
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = 0;
    ib.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = img.image;
    ib.subresourceRange = {aspect, 0, 1, 0, 1};
    m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    return true;
}

void GsVulkanBackend::destroyImage(GpuImage &img)
{
    if (img.view) m_dt.vkDestroyImageView(m_device, img.view, nullptr);
    if (img.image) m_dt.vkDestroyImage(m_device, img.image, nullptr);
    if (img.memory) m_dt.vkFreeMemory(m_device, img.memory, nullptr);
    img = GpuImage{};
}

bool GsVulkanBackend::createDeviceObjects()
{
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = m_queueFamily;
    if (m_dt.vkCreateCommandPool(m_device, &cpi, nullptr, &m_cmdPool) != VK_SUCCESS)
        return false;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = m_cmdPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (Slot &sl : m_slots)
        if (m_dt.vkAllocateCommandBuffers(m_device, &cai, &sl.cmd) != VK_SUCCESS || m_dt.vkCreateFence(m_device, &fci, nullptr, &sl.fence) != VK_SUCCESS)
            return false;
    m_cmd = m_slots[0].cmd;
    m_fence = m_slots[0].fence;
    void *mapped = nullptr;
    if (!createBuffer(kRingSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_ring, m_ringMem, &mapped))
        return false;
    m_ringPtr = static_cast<uint8_t *>(mapped);

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (m_dt.vkCreateSampler(m_device, &sci, nullptr, &m_sampler) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayoutBinding bind{};
    bind.binding = 0;
    bind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bind.descriptorCount = 1;
    bind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bind.pImmutableSamplers = &m_sampler;
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1;
    dli.pBindings = &bind;
    if (m_dt.vkCreateDescriptorSetLayout(m_device, &dli, nullptr, &m_setLayout) != VK_SUCCESS)
        return false;
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConsts)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    const VkDescriptorSetLayout layouts[2] = {m_setLayout, m_setLayout};
    pli.setLayoutCount = 2;
    pli.pSetLayouts = layouts;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (m_dt.vkCreatePipelineLayout(m_device, &pli, nullptr, &m_pipeLayout) != VK_SUCCESS)
        return false;

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(kGsVertSpv);
    smi.pCode = kGsVertSpv;
    if (m_dt.vkCreateShaderModule(m_device, &smi, nullptr, &m_vs) != VK_SUCCESS)
        return false;
    smi.codeSize = m_dualSrc ? sizeof(kGsFragDualSpv) : sizeof(kGsFragSpv);
    smi.pCode = m_dualSrc ? kGsFragDualSpv : kGsFragSpv;
    if (m_dt.vkCreateShaderModule(m_device, &smi, nullptr, &m_fs) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 4096;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    if (m_dt.vkCreateDescriptorPool(m_device, &dpi, nullptr, &m_descPool) != VK_SUCCESS)
        return false;

    // 1x1 texture bound by untextured draws.
    if (!createImage(m_dummyTex, 1, 1, VK_FORMAT_R8G8B8A8_UINT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_descPool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &m_setLayout;
    if (m_dt.vkAllocateDescriptorSets(m_device, &dai, &m_dummySet) != VK_SUCCESS)
        return false;
    VkDescriptorImageInfo dii{VK_NULL_HANDLE, m_dummyTex.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_dummySet;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    m_dt.vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
    if (m_dt.vkAllocateDescriptorSets(m_device, &dai, &m_dstSet) != VK_SUCCESS || !ensureDstImage(1024u))
        return false;
    if (!m_cpuDecode && !createComputeObjects())
    {
        std::fprintf(stderr, "[gs:vk] GPU texture decode unavailable, decoding on the CPU\n");
        m_cpuDecode = true;
    }
    if (m_gpuPresent && (m_cpuDecode || !m_displayPipe))
        m_gpuPresent = false;
    if (m_gpuPresent)
    {
        // Display images, the display pass's counters and its "done" timeline semaphore.
        bool ok = true;
        for (GpuImage &img : m_presentImg)
            ok = ok && createImage(img, kPresentW, kPresentH, VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        ok = ok && createBuffer(64u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                m_dispCount, m_dispCountMem, nullptr);
        VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &tci;
        ok = ok && m_dt.vkCreateSemaphore(m_device, &sci, nullptr, &m_readySem) == VK_SUCCESS;
        if (m_checkPresent)
            for (Slot &sl : m_slots)
            {
                void *mapped = nullptr;
                ok = ok && createBuffer(kPresentW * kPresentH * 4u, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, sl.checkBuf, sl.checkMem, &mapped,
                                        &sl.checkCoherent);
                sl.checkPtr = static_cast<uint8_t *>(mapped);
            }
        if (!ok)
        {
            std::fprintf(stderr, "[gs:vk] GPU presentation unavailable, presenting through the CPU\n");
            m_gpuPresent = false;
        }
    }
    submitAndWait();
    return true;
}

bool GsVulkanBackend::ensureDstImage(uint32_t width)
{
    if (m_dstImg.image && m_dstImg.width >= width)
        return true;
    submitAndWait(); // the descriptor set may be in use
    if (m_dstImg.image)
        destroyImage(m_dstImg);
    if (!createImage(m_dstImg, std::max(width, 1024u), kTargetHeight, VK_FORMAT_R8G8B8A8_UINT,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    VkDescriptorImageInfo dii{VK_NULL_HANDLE, m_dstImg.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_dstSet;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    m_dt.vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
    return true;
}

void GsVulkanBackend::beginCmd()
{
    if (m_cmdOpen)
        return;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    m_dt.vkBeginCommandBuffer(m_cmd, &bi);
    m_cmdOpen = true;
    m_curPipeline = VK_NULL_HANDLE;
    // Everything in the previous command buffer happens before anything in this one.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void GsVulkanBackend::endRendering()
{
    if (!m_rendering)
        return;
    m_dt.vkCmdEndRendering(m_cmd);
    m_rendering = false;
    m_curColor = m_curDepth = nullptr;
}

void GsVulkanBackend::barrier()
{
    beginCmd();
    endRendering();
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// Submits the current command buffer (if any) and waits for everything to complete.
void GsVulkanBackend::submitAndWait()
{
    if (m_cmdOpen)
        queueSubmit();
    {
        ScopeTimer timer(m_iv.waitNs);
        ++m_iv.waits;
        completeSlot(m_cur ^ 1u);
        completeSlot(m_cur);
    }
    m_ringOffset = 0;
    m_poolUsed = 0;
}

// Submits the current command buffer without waiting for it, and continues in the other slot
// (waiting only for that slot's previous submission).
void GsVulkanBackend::submitAsync()
{
    if (m_cmdOpen)
        queueSubmit();
    m_cur ^= 1u;
    {
        ScopeTimer timer(m_iv.waitNs);
        completeSlot(m_cur);
    }
    m_cmd = m_slots[m_cur].cmd;
    m_fence = m_slots[m_cur].fence;
    m_compPool = m_slots[m_cur].compPool;
    m_ringOffset = 0;
    m_poolUsed = 0;
}

// Ends and submits the current command buffer. A display pass in it signals m_readySem and is
// published (to the host) once submitted.
void GsVulkanBackend::queueSubmit()
{
    endRendering();
    m_dt.vkEndCommandBuffer(m_cmd);
    Slot &sl = m_slots[m_cur];
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_cmd;
    VkTimelineSemaphoreSubmitInfo tsi{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    if (sl.gpuImage >= 0)
    {
        sl.gpuValue = ++m_readyValue;
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &sl.gpuValue;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &m_readySem;
        si.pNext = &tsi;
    }
    VkResult r;
    if (m_queueMutex)
    {
        std::lock_guard<std::mutex> qlock(*m_queueMutex);
        r = m_dt.vkQueueSubmit(m_queue, 1, &si, m_fence);
    }
    else
        r = m_dt.vkQueueSubmit(m_queue, 1, &si, m_fence);
    if (r != VK_SUCCESS)
        std::fprintf(stderr, "[gs:vk] vkQueueSubmit failed (%d)\n", static_cast<int>(r));
    sl.pending = true;
    sl.serial = m_submitSerial++;
    m_cmdOpen = false;
    ++m_statSubmits;
    if (sl.gpuImage >= 0)
    {
        std::lock_guard<std::mutex> lock(m_gpuFrameMutex);
        m_pubImage = sl.gpuImage;
        m_pubW = sl.gpuW;
        m_pubH = sl.gpuH;
        m_pubValue = sl.gpuValue;
        m_pubGpu = true;
        m_pubNs = nowNs();
        sl.gpuImage = -1;
    }
}

void GsVulkanBackend::completeSlot(uint32_t i)
{
    Slot &sl = m_slots[i];
    if (!sl.pending)
        return;
    m_dt.vkWaitForFences(m_device, 1, &sl.fence, VK_TRUE, UINT64_MAX);
    m_dt.vkResetFences(m_device, 1, &sl.fence);
    m_dt.vkResetCommandBuffer(sl.cmd, 0);
    sl.pending = false;
    m_completedSerial = std::max(m_completedSerial, sl.serial + 1u);
    if (sl.compPool)
        m_dt.vkResetDescriptorPool(m_device, sl.compPool, 0);
    for (GpuImage &img : sl.defImages)
        destroyImage(img);
    sl.defImages.clear();
    if (!sl.defSets.empty())
    {
        m_dt.vkFreeDescriptorSets(m_device, m_descPool, static_cast<uint32_t>(sl.defSets.size()), sl.defSets.data());
        sl.defSets.clear();
    }
    if (sl.hasPresent)
    {
        // The display pages as they were at the flip, over the rest of GS memory.
        sl.hasPresent = false;
        if (!sl.presentCoherent)
        {
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            r.memory = sl.presentMem;
            r.size = VK_WHOLE_SIZE;
            m_dt.vkInvalidateMappedMemoryRanges(m_device, 1, &r);
        }
        std::vector<uint8_t> snap(m_vram, m_vram + m_vramSize);
        for (uint16_t p : sl.presentPages)
            std::memcpy(snap.data() + static_cast<size_t>(p) * 8192u, sl.presentPtr + static_cast<size_t>(p) * 8192u, 8192u);
        if (sl.hasCheck)
            checkPresent(sl, snap);
        std::lock_guard<std::mutex> plock(m_presentMutex);
        m_snapLatest.swap(snap);
        m_snapRequest = sl.presentReq;
        m_haveSnap = true;
        m_lastFlipNs.store(nowNs(), std::memory_order_relaxed);
    }
}

// Room for an operation that records several dependent allocations: switch command buffers
// first rather than in the middle of it.
void GsVulkanBackend::reserve(uint32_t ringBytes, uint32_t sets)
{
    if (m_ringOffset + ringBytes + 4096u > kRingHalf || m_poolUsed + sets > kCompSets)
        submitAsync();
}

uint32_t GsVulkanBackend::ringAlloc(uint32_t size, uint32_t align)
{
    uint32_t off = (m_ringOffset + align - 1u) & ~(align - 1u);
    if (off + size > kRingHalf)
    {
        submitAsync();
        off = 0;
    }
    m_ringOffset = off + size;
    beginCmd();
    return off + m_cur * kRingHalf;
}

// key: bit0 blend, bits1-2 op, bits3-4 dst factor, bits5-8 colour write mask, bit9 depth attachment,
// bit10 blend factor in source alpha (no dual-source blending), bit11 source factor DST_COLOR,
// bit12 source factor DST_ALPHA
VkPipeline GsVulkanBackend::getPipeline(uint32_t key)
{
    auto it = m_pipelines.find(key);
    if (it != m_pipelines.end())
        return it->second;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = m_vs;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = m_fs;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vb{0, sizeof(GpuVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription va[4] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GpuVertex, x)},
        {1, 0, VK_FORMAT_R8G8B8A8_UINT, offsetof(GpuVertex, r)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GpuVertex, s)},
        {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(GpuVertex, fog)},
    };
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = 4;
    vi.pVertexAttributeDescriptions = va;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = m_depthClamp ? VK_TRUE : VK_FALSE;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendAttachmentState cba{};
    static const VkBlendOp ops[3] = {VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT};
    static const VkBlendFactor dsts[4] = {VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC1_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA};
    static const VkBlendFactor dstsAlpha[4] = {VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA};
    cba.blendEnable = (key & 1u) ? VK_TRUE : VK_FALSE;
    cba.colorBlendOp = ops[(key >> 1) & 3u];
    cba.srcColorBlendFactor = (key & 0x800u) ? VK_BLEND_FACTOR_DST_COLOR : (key & 0x1000u) ? VK_BLEND_FACTOR_DST_ALPHA : VK_BLEND_FACTOR_ONE;
    cba.dstColorBlendFactor = (key & 0x400u) ? dstsAlpha[(key >> 3) & 3u] : dsts[(key >> 3) & 3u];
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.colorWriteMask = (key >> 5) & 0xFu;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                                  VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 5;
    dy.pDynamicStates = dyn;
    const VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo pri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    pri.colorAttachmentCount = 1;
    pri.pColorAttachmentFormats = &colorFormat;
    pri.depthAttachmentFormat = (key & 0x200u) ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED;
    VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpi.pNext = &pri;
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vp;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dy;
    gpi.layout = m_pipeLayout;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (m_dt.vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &gpi, nullptr, &pipe) != VK_SUCCESS)
        std::fprintf(stderr, "[gs:vk] pipeline creation failed (key %x)\n", key);
    m_pipelines[key] = pipe;
    return pipe;
}

// ------------------------------------------------------------------------------------------
// Coherency between GS memory and the GPU targets
// ------------------------------------------------------------------------------------------

GsVulkanBackend::Target &GsVulkanBackend::getTarget(uint32_t fbp, uint32_t fbw, uint8_t psm)
{
    fbw = std::max<uint32_t>(fbw, 1u);
    for (Target *t : m_targets)
        if (t->fbp == fbp && t->fbw == fbw && t->psm == psm)
            return *t;
    Target *t = new Target();
    t->fbp = fbp;
    t->fbw = fbw;
    t->psm = psm;
    t->depth = isDepthPsm(psm);
    const uint32_t w = fbw * 64u;
    if (t->depth)
        createImage(t->img, w, kTargetHeight, VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
    else
        createImage(t->img, w, kTargetHeight, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT);
    const uint32_t groups = kTargetHeight / pageDims(psm).h;
    t->stale = groups >= 64u ? ~0ull : ((1ull << groups) - 1ull);
    t->allPages = groupPages(*t, t->stale);
    m_targets.push_back(t);
    if (m_stats)
        std::fprintf(stderr, "[gs:vk] target fbp=0x%x fbw=%u psm=0x%x (%ux%u)\n", fbp, fbw, psm, w, kTargetHeight);
    return *t;
}

uint64_t GsVulkanBackend::groupsFor(const Target &t, const PageSet &pages) const
{
    if (!t.allPages.intersects(pages))
        return 0;
    const PageDims d = pageDims(t.psm);
    const uint32_t ppr = std::max<uint32_t>(1u, t.fbw * 64u / d.w);
    const uint32_t groups = kTargetHeight / d.h;
    uint64_t out = 0;
    for (uint32_t g = 0; g < groups; ++g)
    {
        const uint32_t base = t.fbp + g * ppr;
        for (uint32_t c = 0; c < ppr; ++c)
            if (pages.test((base + c) & 511u))
            {
                out |= 1ull << g;
                break;
            }
    }
    return out;
}

PageSet GsVulkanBackend::groupPages(const Target &t, uint64_t groups) const
{
    PageSet s;
    const PageDims d = pageDims(t.psm);
    const uint32_t ppr = std::max<uint32_t>(1u, t.fbw * 64u / d.w);
    for (uint32_t g = 0; g < 64u; ++g)
        if (groups & (1ull << g))
            for (uint32_t c = 0; c < ppr; ++c)
                s.set((t.fbp + g * ppr + c) & 511u);
    return s;
}

void GsVulkanBackend::bumpPages(const PageSet &pages)
{
    ++m_serial;
    pages.forEach([&](uint32_t p) { m_pageSerial[p] = m_serial; });
}

// GS memory pages about to be read or written on the CPU: bring in newer render-target data.
void GsVulkanBackend::ensureVramCurrent(const PageSet &pages)
{
    for (size_t i = 0; i < m_targets.size(); ++i)
    {
        Target &t = *m_targets[i];
        if (t.dirty && t.dirtyPages.intersects(pages))
            download(t);
    }
    if (gpuMem())
    {
        const PageSet newer = m_mirrorNewer & pages;
        if (newer.any())
            readbackMirror(newer);
    }
}

// Mirror pages about to be read on the GPU: render targets with newer data written into it,
// pages GS memory changed copied up.
void GsVulkanBackend::ensureMirrorCurrent(const PageSet &pages)
{
    for (size_t i = 0; i < m_targets.size(); ++i)
    {
        Target &t = *m_targets[i];
        if (t.dirty && t.dirtyPages.intersects(pages))
            download(t);
    }
    syncMirrorPages(pages);
}

bool GsVulkanBackend::ensureReadback(VkDeviceSize bytes)
{
    if (bytes <= m_readbackSize)
        return true;
    submitAndWait();
    if (m_readback) m_dt.vkDestroyBuffer(m_device, m_readback, nullptr);
    if (m_readbackMem) m_dt.vkFreeMemory(m_device, m_readbackMem, nullptr);
    m_readback = VK_NULL_HANDLE;
    m_readbackMem = VK_NULL_HANDLE;
    const VkDeviceSize size = std::max<VkDeviceSize>(bytes, 8u << 20);
    void *mapped = nullptr;
    if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, m_readback, m_readbackMem,
                      &mapped, &m_readbackCoherent))
    {
        std::fprintf(stderr, "[gs:vk] readback buffer allocation failed\n");
        m_readbackSize = 0;
        return false;
    }
    m_readbackPtr = static_cast<uint8_t *>(mapped);
    m_readbackSize = size;
    return true;
}

// Copies mirror pages holding render-target data back into GS memory (one GPU round trip;
// plain 8 KiB pages, already in GS memory order).
void GsVulkanBackend::readbackMirror(const PageSet &pages)
{
    ScopeTimer timer(m_iv.downloadNs);
    ++m_iv.downloads;
    ++m_statDownloads;
    if (m_stats)
    {
        char k[64];
        std::snprintf(k, sizeof(k), "%s (pages)", m_why);
        ++m_statWhy[k];
    }
    if (!ensureReadback(GSMem::MEMORY_SIZE))
        return;
    std::vector<VkBufferCopy> copies;
    pages.forEach([&](uint32_t p) { copies.push_back({static_cast<VkDeviceSize>(p) * 8192u, static_cast<VkDeviceSize>(copies.size()) * 8192u, 8192u}); });
    barrier();
    m_dt.vkCmdCopyBuffer(m_cmd, m_vramBuf, m_readback, static_cast<uint32_t>(copies.size()), copies.data());
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    submitAndWait();
    if (!m_readbackCoherent)
    {
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = m_readbackMem;
        r.size = VK_WHOLE_SIZE;
        m_dt.vkInvalidateMappedMemoryRanges(m_device, 1, &r);
    }
    m_statDownloadPx += copies.size() * 2048u;
    for (const VkBufferCopy &c : copies)
        std::memcpy(m_vram + c.srcOffset, m_readbackPtr + c.dstOffset, 8192u);
    bumpPages(pages);
    for (uint32_t p = 0; p < 512u; ++p)
        if (pages.test(p))
            m_mirrorSerial[p] = m_pageSerial[p]; // GS memory and mirror agree again
    m_mirrorNewer &= ~pages;
}

void GsVulkanBackend::markCpuWrite(const PageSet &pages)
{
    bumpPages(pages);
    for (Target *t : m_targets)
        t->stale |= groupsFor(*t, pages);
}

void GsVulkanBackend::download(Target &t)
{
    if (!t.dirty)
        return;
    if (gpuMem())
    {
        // Into the GPU mirror of GS memory (GS memory itself follows only when the CPU reads it).
        overlayTarget(t);
        m_mirrorNewer |= t.dirtyPages;
        t.dirty = false;
        t.dirtyPages.reset();
        return;
    }
    ScopeTimer timer(m_iv.downloadNs);
    ++m_iv.downloads;
    const int x0 = std::max(t.dx0, 0), y0 = std::max(t.dy0, 0);
    const int x1 = std::min<int>(t.dx1, static_cast<int>(t.img.width) - 1), y1 = std::min<int>(t.dy1, static_cast<int>(t.img.height) - 1);
    t.dirty = false;
    t.dirtyPages.reset();
    if (x1 < x0 || y1 < y0)
        return;
    const uint32_t w = static_cast<uint32_t>(x1 - x0 + 1), h = static_cast<uint32_t>(y1 - y0 + 1);
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4u;
    if (bytes > m_readbackSize)
    {
        submitAndWait();
        if (m_readback) m_dt.vkDestroyBuffer(m_device, m_readback, nullptr);
        if (m_readbackMem) m_dt.vkFreeMemory(m_device, m_readbackMem, nullptr);
        m_readback = VK_NULL_HANDLE;
        m_readbackMem = VK_NULL_HANDLE;
        const VkDeviceSize size = std::max<VkDeviceSize>(bytes, 8u << 20);
        void *mapped = nullptr;
        if (!createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, m_readback, m_readbackMem, &mapped, &m_readbackCoherent))
        {
            std::fprintf(stderr, "[gs:vk] readback buffer allocation failed\n");
            m_readbackSize = 0;
            return;
        }
        m_readbackPtr = static_cast<uint8_t *>(mapped);
        m_readbackSize = size;
    }
    barrier();
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = w;
    region.bufferImageHeight = h;
    region.imageSubresource = {static_cast<VkImageAspectFlags>(t.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
    region.imageOffset = {x0, y0, 0};
    region.imageExtent = {w, h, 1};
    m_dt.vkCmdCopyImageToBuffer(m_cmd, t.img.image, VK_IMAGE_LAYOUT_GENERAL, m_readback, 1, &region);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    submitAndWait();
    if (!m_readbackCoherent)
    {
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = m_readbackMem;
        r.size = VK_WHOLE_SIZE;
        m_dt.vkInvalidateMappedMemoryRanges(m_device, 1, &r);
    }

    ++m_statDownloads;
    m_statDownloadPx += static_cast<uint64_t>(w) * h;
    if (m_stats)
    {
        char k[96];
        std::snprintf(k, sizeof(k), "%s fbp=%x psm=%x", m_why, t.fbp, t.psm);
        ++m_statWhy[k];
    }
    const WriteFn write = writeFn(t.psm);
    const uint32_t bp = t.fbp << 5;
    const uint32_t *src = reinterpret_cast<const uint32_t *>(m_readbackPtr);
    if (t.depth)
    {
        const double inv = 1.0 / depthScale(t.psm);
        const uint32_t zmax = depthMax(t.psm);
        const float *fsrc = reinterpret_cast<const float *>(m_readbackPtr);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
            {
                const double z = std::floor(static_cast<double>(fsrc[y * w + x]) * inv + 0.5);
                const uint32_t zi = z <= 0.0 ? 0u : (z >= static_cast<double>(zmax) ? zmax : static_cast<uint32_t>(z));
                write(m_vram, bp, t.fbw, static_cast<uint32_t>(x0) + x, static_cast<uint32_t>(y0) + y, zi);
            }
    }
    else if (t.psm == GS_PSM_CT16 || t.psm == GS_PSM_CT16S)
    {
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                write(m_vram, bp, t.fbw, static_cast<uint32_t>(x0) + x, static_cast<uint32_t>(y0) + y, rgba8888To5551(src[y * w + x]));
    }
    else
    {
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                write(m_vram, bp, t.fbw, static_cast<uint32_t>(x0) + x, static_cast<uint32_t>(y0) + y, src[y * w + x]);
    }
    PageSet pages;
    addRectPages(pages, bp, t.fbw, t.psm, x0, y0, x1, y1);
    bumpPages(pages); // cached textures of these pages are out of date
}

// Everything into GS memory (render targets, and the mirror pages holding their data).
void GsVulkanBackend::downloadAll(bool includeDepth)
{
    for (size_t i = 0; i < m_targets.size(); ++i)
        if (includeDepth || !m_targets[i]->depth)
            download(*m_targets[i]);
    if (gpuMem() && m_mirrorNewer.any())
        readbackMirror(m_mirrorNewer);
}

// Uploads the stale row groups of `t` that intersect rows [y0, y1] (the rest stay stale: nothing
// reads them until a later draw needs them, and only rows a draw touched are ever downloaded).
void GsVulkanBackend::uploadStale(Target &t, int y0, int y1)
{
    const PageDims d = pageDims(t.psm);
    if (!t.stale || y1 < y0)
        return;
    const uint32_t g0 = static_cast<uint32_t>(std::max(y0, 0)) / d.h;
    const uint32_t g1 = std::min<uint32_t>(static_cast<uint32_t>(std::max(y1, 0)) / d.h, 63u);
    const uint64_t range = (g1 >= 63u ? ~0ull : ((1ull << (g1 + 1u)) - 1ull)) & ~((1ull << g0) - 1ull);
    const uint64_t stale = t.stale & range;
    if (!stale)
        return;
    m_why = "upload";
    if (gpuMem())
        ensureMirrorCurrent(groupPages(t, stale));
    else
        ensureVramCurrent(groupPages(t, stale));
    ScopeTimer uploadTimer(m_iv.uploadNs);
    ++m_iv.uploads;
    t.stale &= ~stale;
    const uint32_t groups = kTargetHeight / d.h;
    if (gpuMem())
    {
        for (uint32_t g = 0; g < groups;)
        {
            if (!(stale & (1ull << g)))
            {
                ++g;
                continue;
            }
            uint32_t g1 = g;
            while (g1 + 1u < groups && (stale & (1ull << (g1 + 1u))))
                ++g1;
            uploadFromMirror(t, g * d.h, (g1 - g + 1u) * d.h);
            g = g1 + 1u;
        }
        return;
    }
    const uint32_t w = t.img.width;
    const ReadFn read = readFn(t.psm);
    const uint32_t bp = t.fbp << 5;
    uint32_t g = 0;
    while (g < groups)
    {
        if (!(stale & (1ull << g)))
        {
            ++g;
            continue;
        }
        uint32_t g1 = g;
        while (g1 + 1u < groups && (stale & (1ull << (g1 + 1u))))
            ++g1;
        const uint32_t y0 = g * d.h, rows = (g1 - g + 1u) * d.h;
        const uint32_t off = ringAlloc(w * rows * 4u, 16u);
        uint32_t *dst = reinterpret_cast<uint32_t *>(m_ringPtr + off);
        if (t.depth)
        {
            const double scale = depthScale(t.psm);
            float *fd = reinterpret_cast<float *>(dst);
            for (uint32_t y = 0; y < rows; ++y)
                for (uint32_t x = 0; x < w; ++x)
                    fd[y * w + x] = static_cast<float>(static_cast<double>(read(m_vram, bp, t.fbw, x, y0 + y)) * scale);
        }
        else
        {
            for (uint32_t y = 0; y < rows; ++y)
                for (uint32_t x = 0; x < w; ++x)
                {
                    uint32_t v = read(m_vram, bp, t.fbw, x, y0 + y);
                    if (t.psm == GS_PSM_CT16 || t.psm == GS_PSM_CT16S)
                        v = rgba5551To8888(v);
                    else if (t.psm == GS_PSM_CT24)
                        v = (v & 0x00FFFFFFu) | 0x80000000u;
                    dst[y * w + x] = v;
                }
        }
        barrier();
        VkBufferImageCopy region{};
        region.bufferOffset = off;
        region.bufferRowLength = w;
        region.bufferImageHeight = rows;
        region.imageSubresource = {static_cast<VkImageAspectFlags>(t.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
        region.imageOffset = {0, static_cast<int32_t>(y0), 0};
        region.imageExtent = {w, rows, 1};
        m_dt.vkCmdCopyBufferToImage(m_cmd, m_ring, t.img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        barrier();
        ++t.version;
        ++m_statUploads;
        m_statUploadRows += rows;
        g = g1 + 1u;
    }
}

void GsVulkanBackend::prepareDrawTarget(Target &t, const PageSet &drawPages, int y0, int y1)
{
    for (size_t i = 0; i < m_targets.size(); ++i)
    {
        Target &o = *m_targets[i];
        if (&o == &t)
            continue;
        const uint64_t g = groupsFor(o, drawPages);
        if (!g && !(o.dirty && o.dirtyPages.intersects(drawPages)))
            continue;
        if (o.dirty && o.dirtyPages.intersects(drawPages))
        {
            m_why = "overlap";
            download(o);
        }
        o.stale |= g;
    }
    uploadStale(t, y0, y1);
}

// GS memory the presentation code may read for this request (display buffers, the preferred
// source, and the context frames it falls back to when display buffer 0 is empty).
PageSet GsVulkanBackend::displayPages(const GSPresentationRequest &request)
{
    PageSet pages;
    auto addFrame = [&](uint32_t fbp, uint32_t fbw, uint8_t psm, uint32_t oy, uint32_t height)
    {
        fbw = std::max<uint32_t>(fbw, 1u);
        addRectPages(pages, fbp << 5, fbw, psm, 0, 0, static_cast<int>(fbw * 64u) - 1, static_cast<int>(std::min<uint32_t>(oy + height, 2048u)) - 1);
    };
    const uint64_t disp[2][2] = {{request.dispfb1, request.display1}, {request.dispfb2, request.display2}};
    bool fbpZero = false;
    for (int i = 0; i < 2; ++i)
    {
        if (!((request.pmode >> i) & 1u))
            continue;
        const uint64_t *d = disp[i];
        const uint32_t fbp = static_cast<uint32_t>(d[0] & 0x1FFu), fbw = static_cast<uint32_t>((d[0] >> 9) & 0x3Fu);
        const uint8_t psm = static_cast<uint8_t>((d[0] >> 15) & 0x1Fu);
        const uint32_t oy = static_cast<uint32_t>((d[0] >> 43) & 0x7FFu);
        const uint32_t dh = static_cast<uint32_t>((d[1] >> 44) & 0x7FFu) + 1u;
        addFrame(fbp, fbw, psm, oy, dh >= 64u ? std::min<uint32_t>(dh, 512u) : 448u);
        fbpZero = fbpZero || fbp == 0u;
    }
    if (request.hasPreferredSource)
        addFrame(request.preferredSource.fbp, request.preferredSource.fbw, request.preferredSource.psm, 0u, 512u);
    if (fbpZero)
        for (const GSFrameReg &f : request.contextFrames)
            addFrame(f.fbp, f.fbw, f.psm, 0u, 512u);
    return pages;
}

uint64_t GsVulkanBackend::pagesHash(const std::vector<uint16_t> &pages) const
{
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (uint16_t p : pages)
    {
        const uint64_t *w = reinterpret_cast<const uint64_t *>(m_vram + static_cast<size_t>(p) * 8192u);
        uint64_t a = h ^ p, b = 0, c = 0, d = 0;
        for (uint32_t i = 0; i < 1024u; i += 4u)
        {
            a = (a ^ w[i]) * 0x100000001B3ull;
            b = (b ^ w[i + 1]) * 0xC2B2AE3D27D4EB4Full;
            c = (c ^ w[i + 2]) * 0x165667B19E3779F9ull;
            d = (d ^ w[i + 3]) * 0x27D4EB2F165667C5ull;
        }
        h = (h ^ a ^ (b << 1) ^ (c << 2) ^ (d << 3)) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 31;
    }
    return h;
}

void GsVulkanBackend::dropAll()
{
    submitAndWait();
    for (Target *t : m_targets)
    {
        destroyImage(t->img);
        delete t;
    }
    m_targets.clear();
    for (auto &kv : m_textures)
    {
        destroyImage(kv.second.img);
        m_dt.vkFreeDescriptorSets(m_device, m_descPool, 1, &kv.second.set);
    }
    m_textures.clear();
    for (auto &kv : m_alias)
    {
        destroyImage(kv.second.img);
        m_dt.vkFreeDescriptorSets(m_device, m_descPool, 1, &kv.second.set);
    }
    m_alias.clear();
}

// ------------------------------------------------------------------------------------------
// Drawing
// ------------------------------------------------------------------------------------------

void GsVulkanBackend::appendPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &st = batch.state;
    const GSContext &ctx = st.context;
    const int ofx = ctx.xyoffset.ofx >> 4;
    const int ofy = ctx.xyoffset.ofy >> 4;
    const bool depthNeeded = m_batch.setup.needDepth;
    const double zscale = depthScale(ctx.zbuf.psm);
    Batch &b = m_batch;
    auto bound = [&](float x, float y)
    {
        b.minX = std::min(b.minX, x);
        b.maxX = std::max(b.maxX, x);
        b.minY = std::min(b.minY, y);
        b.maxY = std::max(b.maxY, y);
    };

    if (st.prim.type == GS_PRIM_POINT || st.prim.type == GS_PRIM_LINE || st.prim.type == GS_PRIM_LINESTRIP)
    {
        // One 1x1 quad per pixel, stepped exactly like GSCpuBackend::DrawLine (Bresenham, colour
        // and Z interpolated per step, Z truncated).
        auto pixel = [&](int x, int y, uint8_t r, uint8_t g, uint8_t bl, uint8_t a, uint32_t zi, uint8_t fog)
        {
            GpuVertex q{};
            q.z = depthNeeded ? static_cast<float>(static_cast<double>(zi) * zscale) : 0.0f;
            q.r = r;
            q.g = g;
            q.b = bl;
            q.a = a;
            q.q = 1.0f;
            q.fog = static_cast<float>(fog);
            const float X0 = static_cast<float>(x), Y0 = static_cast<float>(y);
            GpuVertex c[4] = {q, q, q, q};
            c[0].x = X0; c[0].y = Y0;
            c[1].x = X0 + 1.0f; c[1].y = Y0;
            c[2].x = X0; c[2].y = Y0 + 1.0f;
            c[3].x = X0 + 1.0f; c[3].y = Y0 + 1.0f;
            b.verts.push_back(c[0]);
            b.verts.push_back(c[1]);
            b.verts.push_back(c[2]);
            b.verts.push_back(c[1]);
            b.verts.push_back(c[3]);
            b.verts.push_back(c[2]);
            bound(X0, Y0);
            b.maxVA = std::max<uint32_t>(b.maxVA, a);
        };
        const GSVertex &v0 = batch.vertices[0];
        if (st.prim.type == GS_PRIM_POINT)
        {
            pixel(static_cast<int>(v0.x) - ofx, static_cast<int>(v0.y) - ofy, v0.r, v0.g, v0.b, v0.a, static_cast<uint32_t>(v0.z), v0.fog);
            return;
        }
        const GSVertex &v1 = batch.vertices[1];
        int x0 = static_cast<int>(v0.x) - ofx, y0 = static_cast<int>(v0.y) - ofy;
        const int x1 = static_cast<int>(v1.x) - ofx, y1 = static_cast<int>(v1.y) - ofy;
        const int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
        const int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
        if (totalSteps == 0)
            totalSteps = 1;
        auto c8 = [](int v) { return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v); };
        for (int step = 0; step < 4096; ++step)
        {
            const float t = static_cast<float>(step) / static_cast<float>(totalSteps);
            uint8_t r = v1.r, g = v1.g, bl = v1.b, a = v1.a;
            if (st.prim.iip)
            {
                r = c8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
                g = c8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
                bl = c8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
                a = c8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
            }
            const double z = v0.z + (v1.z - v0.z) * t;
            const uint8_t fog = c8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
            pixel(x0, y0, r, g, bl, a, static_cast<uint32_t>(z), fog);
            if (x0 == x1 && y0 == y1)
                break;
            const int e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
        }
        return;
    }

    if (st.prim.type == GS_PRIM_SPRITE)
    {
        const GSVertex &v0 = batch.vertices[0];
        const GSVertex &v1 = batch.vertices[1];
        int x0 = static_cast<int>(v0.x) - ofx, y0 = static_cast<int>(v0.y) - ofy;
        int x1 = static_cast<int>(v1.x) - ofx, y1 = static_cast<int>(v1.y) - ofy;
        if (x0 > x1)
            std::swap(x0, x1);
        if (y0 > y1)
            std::swap(y0, y1);
        const int spanX = std::max(1, x1 - x0), spanY = std::max(1, y1 - y0);
        const float X0 = static_cast<float>(x0), Y0 = static_cast<float>(y0);
        const float X1 = static_cast<float>(x0 + spanX), Y1 = static_cast<float>(y0 + spanY);
        float s0 = 0, t0 = 0, s1 = 0, t1 = 0;
        if (st.prim.tme)
        {
            if (st.prim.fst)
            {
                s0 = static_cast<float>((v0.u >> 4) * 16u);
                t0 = static_cast<float>((v0.v >> 4) * 16u);
                s1 = static_cast<float>((v1.u >> 4) * 16u);
                t1 = static_cast<float>((v1.v >> 4) * 16u);
            }
            else
            {
                const float q0 = std::fabs(v0.q) > 1.0e-8f ? v0.q : 1.0f;
                const float q1 = std::fabs(v1.q) > 1.0e-8f ? v1.q : 1.0f;
                s0 = (v0.s / q0);
                t0 = (v0.t / q0);
                s1 = (v1.s / q1);
                t1 = (v1.t / q1);
            }
        }
        b.maxVA = std::max<uint32_t>(b.maxVA, v1.a);
        const uint32_t zi = static_cast<uint32_t>(v1.z);
        const float z = depthNeeded ? static_cast<float>(static_cast<double>(zi) * zscale) : 0.0f;
        auto mk = [&](float x, float y, float s, float t)
        {
            GpuVertex g{};
            g.x = x;
            g.y = y;
            g.z = z;
            g.r = v1.r;
            g.g = v1.g;
            g.b = v1.b;
            g.a = v1.a;
            g.s = s;
            g.t = t;
            g.q = 1.0f;
            g.fog = static_cast<float>(v1.fog);
            return g;
        };
        const GpuVertex a = mk(X0, Y0, s0, t0), bb = mk(X1, Y0, s1, t0), c = mk(X0, Y1, s0, t1), d = mk(X1, Y1, s1, t1);
        b.verts.push_back(a);
        b.verts.push_back(bb);
        b.verts.push_back(c);
        b.verts.push_back(bb);
        b.verts.push_back(d);
        b.verts.push_back(c);
        bound(X0, Y0);
        bound(X1 - 1.0f, Y1 - 1.0f);
        return;
    }

    const GSVertex &flat = batch.vertices[2];
    for (int i = 0; i < 3; ++i)
    {
        const GSVertex &v = batch.vertices[i];
        GpuVertex g{};
        g.x = v.x - static_cast<float>(ofx);
        g.y = v.y - static_cast<float>(ofy);
        g.z = depthNeeded ? static_cast<float>(v.z * zscale) : 0.0f;
        const GSVertex &cv = st.prim.iip ? v : flat;
        b.maxVA = std::max<uint32_t>(b.maxVA, cv.a);
        g.r = cv.r;
        g.g = cv.g;
        g.b = cv.b;
        g.a = cv.a;
        if (st.prim.fst)
        {
            g.s = static_cast<float>(v.u);
            g.t = static_cast<float>(v.v);
            g.q = 1.0f;
        }
        else
        {
            g.s = v.s;
            g.t = v.t;
            g.q = v.q;
        }
        g.fog = static_cast<float>(v.fog);
        b.verts.push_back(g);
        bound(std::floor(g.x), std::floor(g.y));
        bound(std::ceil(g.x), std::ceil(g.y));
    }
}

void GsVulkanBackend::Submit(const GSPrimitiveBatch &batch)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    ScopeTimer timer(m_iv.submitNs);
    ++m_iv.prims;
    if (!m_vram || batch.vertexCount == 0u)
        return;
    if (primitiveOutsideScissor(batch))
        return;
    const int rounding = std::fegetround();
    if (rounding != FE_TONEAREST)
        std::fesetround(FE_TONEAREST);
    ++m_statPrims;

    const bool sameAsBatch = m_batch.active && std::memcmp(&m_batch.state, &batch.state, sizeof(GSDrawState)) == 0;
    if (!sameAsBatch)
    {
        if (!m_haveLastAnalysed || std::memcmp(&m_lastAnalysed, &batch.state, sizeof(GSDrawState)) != 0)
        {
            m_lastAnalysed = batch.state;
            m_lastSetup = analyseState(batch.state);
            m_haveLastAnalysed = true;
        }
        const DrawSetup &setup = m_lastSetup;
        if (setup.fallback >= 0)
        {
            flushBatch();
            cpuDraw(batch, setup.fallback);
        }
        else if (!setup.skip)
        {
            flushBatch();
            m_batch.active = true;
            m_batch.state = batch.state;
            m_batch.setup = setup;
            appendPrimitive(batch);
        }
    }
    else
    {
        appendPrimitive(batch);
        if (m_batch.verts.size() >= kMaxBatchVertices)
            flushBatch();
    }
    if (rounding != FE_TONEAREST)
        std::fesetround(rounding);
}

void GsVulkanBackend::cpuDraw(const GSPrimitiveBatch &batch, int reason)
{
    ++m_statFallback[reason];
    const GSDrawState &st = batch.state;
    const GSContext &ctx = st.context;
    {
        static const int s_log = std::getenv("PS2_GS_VK_LOGFB") ? std::atoi(std::getenv("PS2_GS_VK_LOGFB")) : 0;
        static int s_logged[FB_COUNT]{};
        if (s_logged[reason] < s_log)
        {
            ++s_logged[reason];
            std::fprintf(stderr, "[gs:vk] cpu %s: type=%d tme=%d abe=%d fbp=%x fbw=%u psm=%x zbp=%x zpsm=%x zmsk=%d fbmsk=%08x alpha=%llx test=%llx tex=%x/%u/%x tfx=%d tcc=%d fba=%d\n",
                         kFallbackNames[reason], st.prim.type, st.prim.tme, st.prim.abe, ctx.frame.fbp, ctx.frame.fbw, ctx.frame.psm, ctx.zbuf.zbp,
                         ctx.zbuf.psm, ctx.zbuf.zmask, ctx.frame.fbmsk, (unsigned long long)ctx.alpha, (unsigned long long)ctx.test, ctx.tex0.tbp0,
                         ctx.tex0.tbw, ctx.tex0.psm, ctx.tex0.tfx, ctx.tex0.tcc, int(ctx.fba & 1u));
        }
    }
    const int ofx = ctx.xyoffset.ofx >> 4, ofy = ctx.xyoffset.ofy >> 4;
    const uint32_t n = st.prim.type == GS_PRIM_POINT ? 1u : (st.prim.type == GS_PRIM_LINE || st.prim.type == GS_PRIM_LINESTRIP || st.prim.type == GS_PRIM_SPRITE) ? 2u : 3u;
    float mnx = 1e30f, mny = 1e30f, mxx = -1e30f, mxy = -1e30f;
    for (uint32_t i = 0; i < n && i < batch.vertexCount; ++i)
    {
        mnx = std::min(mnx, batch.vertices[i].x - ofx);
        mxx = std::max(mxx, batch.vertices[i].x - ofx);
        mny = std::min(mny, batch.vertices[i].y - ofy);
        mxy = std::max(mxy, batch.vertices[i].y - ofy);
    }
    const int x0 = std::max<int>(static_cast<int>(std::floor(mnx)) - 1, ctx.scissor.x0);
    const int y0 = std::max<int>(static_cast<int>(std::floor(mny)) - 1, ctx.scissor.y0);
    const int x1 = std::min<int>(static_cast<int>(std::ceil(mxx)) + 1, ctx.scissor.x1);
    const int y1 = std::min<int>(static_cast<int>(std::ceil(mxy)) + 1, ctx.scissor.y1);
    PageSet fb, z, tex;
    const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
    addRectPages(fb, ctx.frame.fbp << 5, fbw, ctx.frame.psm, x0, y0, x1, y1);
    addRectPages(z, ctx.zbuf.zbp << 5, fbw, ctx.zbuf.psm, x0, y0, x1, y1);
    if (st.prim.tme)
        addRectPages(tex, ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0, 0, 1023, 1023);
    ensureVramCurrent(fb | z | tex);
    m_cpu.Submit(batch);
    markCpuWrite(fb | z);
}

// ------------------------------------------------------------------------------------------
// GPU mirror of GS memory: texture decode without GPU->CPU round trips
// ------------------------------------------------------------------------------------------

namespace
{
    // Table index in m_lutOffset for a PSM, and its page size.
    int lutIndex(uint8_t psm)
    {
        switch (psm)
        {
        case GS_PSM_CT32: case GS_PSM_CT24: case GS_PSM_T8H: case GS_PSM_T4HL: case GS_PSM_T4HH: return 0;
        case GS_PSM_CT16: return 1;
        case GS_PSM_CT16S: return 2;
        case GS_PSM_T8: return 3;
        case GS_PSM_T4: return 4;
        case GS_PSM_Z32: case GS_PSM_Z24: return 5;
        case GS_PSM_Z16: return 6;
        case GS_PSM_Z16S: return 7;
        default: return -1;
        }
    }

    struct DecodePush
    {
        uint32_t kind, tableOff, pw, ph, bp, bw, w, h, texa, slot;
    };
    struct OverlayPush
    {
        uint32_t kind, tableOff, pw, ph, bp, bw, x0, y0, w, h;
    };
}

int GsVulkanBackend::decodeKind(uint8_t psm)
{
    switch (psm)
    {
    case GS_PSM_CT32: return 0;
    case GS_PSM_CT24: return 1;
    case GS_PSM_CT16: case GS_PSM_CT16S: return 2;
    case GS_PSM_Z32: return 3;
    case GS_PSM_Z24: return 4;
    case GS_PSM_Z16: case GS_PSM_Z16S: return 5;
    case GS_PSM_T8: return 6;
    case GS_PSM_T8H: return 7;
    case GS_PSM_T4: return 8;
    case GS_PSM_T4HL: return 9;
    case GS_PSM_T4HH: return 10;
    default: return -1;
    }
}

bool GsVulkanBackend::createComputeObjects()
{
    VkPhysicalDeviceProperties props{};
    m_it.vkGetPhysicalDeviceProperties(m_phys, &props);
    m_ssboAlign = std::max<VkDeviceSize>(props.limits.minStorageBufferOffsetAlignment, 16u);

    // Swizzle tables, 16-bit entries back to back.
    static const uint8_t tablePsm[8] = {GS_PSM_CT32, GS_PSM_CT16, GS_PSM_CT16S, GS_PSM_T8, GS_PSM_T4, GS_PSM_Z32, GS_PSM_Z16, GS_PSM_Z16S};
    std::vector<uint16_t> lut;
    for (int i = 0; i < 8; ++i)
    {
        const PageDims d = pageDims(tablePsm[i]);
        const size_t entries = 32u * d.w * d.h;
        m_lutOffset[i] = static_cast<uint32_t>(lut.size());
        const uint16_t *src = static_cast<const uint16_t *>(GSMem::PageTableData(tablePsm[i]));
        if (!src)
            return false;
        lut.insert(lut.end(), src, src + entries);
    }
    if (lut.size() & 1u)
        lut.push_back(0);
    const VkDeviceSize lutBytes = lut.size() * sizeof(uint16_t);
    if (!createBuffer(lutBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_lutBuf, m_lutMem, nullptr) ||
        !createBuffer(GSMem::MEMORY_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vramBuf, m_vramMem, nullptr) ||
        !createBuffer(16u << 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_scratch, m_scratchMem, nullptr))
        return false;
    {
        const uint32_t off = ringAlloc(static_cast<uint32_t>(lutBytes), 16u);
        std::memcpy(m_ringPtr + off, lut.data(), lutBytes);
        VkBufferCopy bc{off, 0, lutBytes};
        m_dt.vkCmdCopyBuffer(m_cmd, m_ring, m_lutBuf, 1, &bc);
        barrier();
    }
    m_mirrorSerial.fill(~0ull);
    {
        void *mapped = nullptr;
        if (!createBuffer(kAlphaSlots * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_alphaRes,
                          m_alphaResMem, &mapped))
            return false;
        m_alphaResPtr = static_cast<uint32_t *>(mapped);
        std::memset(m_alphaResPtr, 0, kAlphaSlots * 4u);
        m_alphaOwner.assign(kAlphaSlots, 0);
    }

    VkDescriptorSetLayoutBinding b[6]{};
    for (uint32_t i = 0; i < 6; ++i)
    {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[2].pImmutableSamplers = &m_sampler;
    b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 6;
    dli.pBindings = b;
    if (m_dt.vkCreateDescriptorSetLayout(m_device, &dli, nullptr, &m_compLayout) != VK_SUCCESS)
        return false;
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &m_compLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (m_dt.vkCreatePipelineLayout(m_device, &pli, nullptr, &m_compPipeLayout) != VK_SUCCESS)
        return false;
    auto makePipe = [&](const uint32_t *code, size_t bytes, VkPipeline &pipe)
    {
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = bytes;
        smi.pCode = code;
        VkShaderModule mod = VK_NULL_HANDLE;
        if (m_dt.vkCreateShaderModule(m_device, &smi, nullptr, &mod) != VK_SUCCESS)
            return false;
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = mod;
        cpi.stage.pName = "main";
        cpi.layout = m_compPipeLayout;
        const bool ok = m_dt.vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe) == VK_SUCCESS;
        m_dt.vkDestroyShaderModule(m_device, mod, nullptr);
        return ok;
    };
    if (!makePipe(kGsDecodeSpv, sizeof(kGsDecodeSpv), m_decodePipe) || !makePipe(kGsOverlaySpv, sizeof(kGsOverlaySpv), m_overlayPipe) ||
        !makePipe(kGsUnswizzleSpv, sizeof(kGsUnswizzleSpv), m_unswizzlePipe))
        return false;
    if (!makePipe(kGsDisplaySpv, sizeof(kGsDisplaySpv), m_displayPipe))
        m_displayPipe = VK_NULL_HANDLE;
    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * kCompSets}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kCompSets}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompSets}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = kCompSets;
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes = ps;
    for (Slot &sl : m_slots)
    {
        if (m_dt.vkCreateDescriptorPool(m_device, &dpi, nullptr, &sl.compPool) != VK_SUCCESS)
            return false;
        void *mapped = nullptr;
        if (!createBuffer(GSMem::MEMORY_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, sl.presentBuf,
                          sl.presentMem, &mapped, &sl.presentCoherent))
            return false;
        sl.presentPtr = static_cast<uint8_t *>(mapped);
    }
    m_compPool = m_slots[m_cur].compPool;
    return true;
}

VkDescriptorSet GsVulkanBackend::computeSet()
{
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = m_compPool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &m_compLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (m_dt.vkAllocateDescriptorSets(m_device, &dai, &set) != VK_SUCCESS)
        std::fprintf(stderr, "[gs:vk] compute descriptor set allocation failed\n");
    ++m_poolUsed;
    VkDescriptorBufferInfo vb{m_vramBuf, 0, VK_WHOLE_SIZE}, lb{m_lutBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2]{};
    for (int i = 0; i < 2; ++i)
    {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    w[0].pBufferInfo = &vb;
    w[1].pBufferInfo = &lb;
    m_dt.vkUpdateDescriptorSets(m_device, 2, w, 0, nullptr);
    return set;
}

// Brings the mirror pages up to date with GS memory where GS memory changed since.
void GsVulkanBackend::syncMirrorPages(const PageSet &pages)
{
    bool any = false;
    (pages & ~m_mirrorNewer).forEach([&](uint32_t p)
    {
        if (m_mirrorSerial[p] == m_pageSerial[p])
            return;
        const uint32_t off = ringAlloc(8192u, 16u);
        std::memcpy(m_ringPtr + off, m_vram + static_cast<size_t>(p) * 8192u, 8192u);
        if (!any)
            barrier(); // after earlier readers of the mirror
        any = true;
        endRendering();
        const VkBufferCopy bc{off, static_cast<VkDeviceSize>(p) * 8192u, 8192u};
        m_dt.vkCmdCopyBuffer(m_cmd, m_ring, m_vramBuf, 1, &bc);
        m_mirrorSerial[p] = m_pageSerial[p];
        m_mirrorGen[p] = ++m_mirrorGenCounter;
        ++m_statMirrorPages;
    });
    if (any)
        barrier();
}

// Target rows [y0, y0 + rows) read out of the mirror (GPU-side upload).
void GsVulkanBackend::uploadFromMirror(Target &t, uint32_t y0, uint32_t rows)
{
    const int li = lutIndex(t.psm);
    const uint32_t w = t.img.width;
    if (li < 0 || static_cast<VkDeviceSize>(w) * rows * 4u > (16u << 20))
        return;
    reserve(0u, 1u);
    VkDescriptorSet set = computeSet();
    VkDescriptorBufferInfo ob{m_scratch, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wd.dstSet = set;
    wd.dstBinding = 5;
    wd.descriptorCount = 1;
    wd.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wd.pBufferInfo = &ob;
    m_dt.vkUpdateDescriptorSets(m_device, 1, &wd, 0, nullptr);
    struct
    {
        uint32_t kind, tableOff, pw, ph, bp, bw, y0, w, h;
    } pc{};
    switch (t.psm)
    {
    case GS_PSM_CT32: pc.kind = 0; break;
    case GS_PSM_CT24: pc.kind = 1; break;
    case GS_PSM_CT16: case GS_PSM_CT16S: pc.kind = 2; break;
    case GS_PSM_Z32: pc.kind = 3; break;
    case GS_PSM_Z24: pc.kind = 4; break;
    default: pc.kind = 5; break;
    }
    const PageDims d = pageDims(t.psm);
    pc.tableOff = m_lutOffset[li];
    pc.pw = d.w;
    pc.ph = d.h;
    pc.bp = t.fbp << 5;
    pc.bw = t.fbw;
    pc.y0 = y0;
    pc.w = w;
    pc.h = rows;
    beginCmd();
    barrier();
    m_dt.vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_unswizzlePipe);
    m_dt.vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compPipeLayout, 0, 1, &set, 0, nullptr);
    m_dt.vkCmdPushConstants(m_cmd, m_compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    m_dt.vkCmdDispatch(m_cmd, (w + 7u) / 8u, (rows + 7u) / 8u, 1);
    barrier();
    VkBufferImageCopy region{};
    region.bufferRowLength = w;
    region.bufferImageHeight = rows;
    region.imageSubresource = {static_cast<VkImageAspectFlags>(t.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
    region.imageOffset = {0, static_cast<int32_t>(y0), 0};
    region.imageExtent = {w, rows, 1};
    m_dt.vkCmdCopyBufferToImage(m_cmd, m_scratch, t.img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    barrier();
    m_curPipeline = VK_NULL_HANDLE;
    ++t.version;
    ++m_statUploads;
    m_statUploadRows += rows;
}

// Writes the dirty rectangle of a render target into the mirror (GPU-side "download").
void GsVulkanBackend::overlayTarget(Target &t)
{
    if (!t.dirty)
        return;
    const int x0 = std::max(t.dx0, 0), y0 = std::max(t.dy0, 0);
    const int x1 = std::min<int>(t.dx1, static_cast<int>(t.img.width) - 1), y1 = std::min<int>(t.dy1, static_cast<int>(t.img.height) - 1);
    if (x1 < x0 || y1 < y0)
        return;
    // Pages of the rectangle not yet in the mirror first (the target only covers part of them).
    syncMirrorPages(t.dirtyPages);
    reserve(0u, 1u);
    const int li = lutIndex(t.psm);
    if (li < 0)
        return;
    VkDescriptorSet set = computeSet();
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, t.img.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set;
    w.dstBinding = 2;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    m_dt.vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
    const PageDims d = pageDims(t.psm);
    OverlayPush pc{};
    switch (t.psm)
    {
    case GS_PSM_CT32: pc.kind = 0; break;
    case GS_PSM_CT24: pc.kind = 1; break;
    case GS_PSM_CT16: case GS_PSM_CT16S: pc.kind = 2; break;
    case GS_PSM_Z32: pc.kind = 3; break;
    case GS_PSM_Z24: pc.kind = 4; break;
    default: pc.kind = 5; break;
    }
    pc.tableOff = m_lutOffset[li];
    pc.pw = d.w;
    pc.ph = d.h;
    pc.bp = t.fbp << 5;
    pc.bw = t.fbw;
    pc.x0 = static_cast<uint32_t>(x0);
    pc.y0 = static_cast<uint32_t>(y0);
    pc.w = static_cast<uint32_t>(x1 - x0 + 1);
    pc.h = static_cast<uint32_t>(y1 - y0 + 1);
    beginCmd();
    endRendering();
    barrier();
    m_dt.vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_overlayPipe);
    m_dt.vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compPipeLayout, 0, 1, &set, 0, nullptr);
    m_dt.vkCmdPushConstants(m_cmd, m_compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    m_dt.vkCmdDispatch(m_cmd, (pc.w + 7u) / 8u, (pc.h + 7u) / 8u, 1);
    barrier();
    m_curPipeline = VK_NULL_HANDLE;
    // The mirror now differs from GS memory on these pages.
    for (uint32_t p = 0; p < 512u; ++p)
        if (t.dirtyPages.test(p))
        {
            m_mirrorSerial[p] = ~0ull;
            m_mirrorGen[p] = ++m_mirrorGenCounter;
        }
    ++m_statOverlays;
}

void GsVulkanBackend::gpuDecode(const GSDrawState &st, Texture &texture, const PageSet &pages, const uint32_t *palette)
{
    const GpuImage &img = texture.img;
    const GSTex0Reg &tex = st.context.tex0;
    ensureMirrorCurrent(pages);
    reserve(2048u, 1u);
    const uint32_t palOff = ringAlloc(1024u, static_cast<uint32_t>(m_ssboAlign));
    if (palette)
        std::memcpy(m_ringPtr + palOff, palette, 1024u);
    VkDescriptorSet set = computeSet();
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, img.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo pb{m_ring, palOff, 1024u};
    VkDescriptorBufferInfo rb{m_alphaRes, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[3]{};
    w[0].sType = w[1].sType = w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = w[1].dstSet = w[2].dstSet = set;
    w[2].dstBinding = 5;
    w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo = &rb;
    w[0].dstBinding = 3;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[0].pImageInfo = &ii;
    w[1].dstBinding = 4;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &pb;
    m_dt.vkUpdateDescriptorSets(m_device, 3, w, 0, nullptr);
    const PageDims d = pageDims(tex.psm);
    DecodePush pc{};
    // Result slot: zeroed now (host-coherent, visible to the submit that follows).
    const uint32_t slot = m_alphaNext;
    m_alphaNext = (m_alphaNext + 1u) % kAlphaSlots;
    m_alphaResPtr[slot] = 0u;
    m_alphaOwner[slot] = ++m_alphaIds;
    texture.alphaSlot = slot;
    texture.alphaId = m_alphaIds;
    texture.alphaSerial = m_submitSerial;
    pc.slot = slot;
    pc.kind = static_cast<uint32_t>(decodeKind(tex.psm));
    pc.tableOff = m_lutOffset[lutIndex(tex.psm)];
    pc.pw = d.w;
    pc.ph = d.h;
    // The CT32-layout formats (T8H, T4HL/HH) use the CT32 page size.
    if (tex.psm == GS_PSM_T8H || tex.psm == GS_PSM_T4HL || tex.psm == GS_PSM_T4HH)
    {
        pc.pw = 64u;
        pc.ph = 32u;
    }
    pc.bp = tex.tbp0;
    pc.bw = tex.tbw;
    pc.w = img.width;
    pc.h = img.height;
    pc.texa = st.texa.ta0 | (static_cast<uint32_t>(st.texa.ta1) << 8) | (st.texa.aem ? 0x10000u : 0u);
    beginCmd();
    endRendering();
    barrier();
    m_dt.vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_decodePipe);
    m_dt.vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compPipeLayout, 0, 1, &set, 0, nullptr);
    m_dt.vkCmdPushConstants(m_cmd, m_compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    m_dt.vkCmdDispatch(m_cmd, (pc.w + 7u) / 8u, (pc.h + 7u) / 8u, 1);
    barrier();
    m_curPipeline = VK_NULL_HANDLE;
}

// Once the decode has executed, the GPU's largest texel alpha replaces the estimate.
void GsVulkanBackend::resolveMaxAlpha(Texture &t)
{
    if (t.alphaSlot == ~0u || m_completedSerial <= t.alphaSerial)
        return;
    if (m_alphaOwner[t.alphaSlot] == t.alphaId)
        t.maxAlpha = static_cast<uint8_t>(std::min<uint32_t>(m_alphaResPtr[t.alphaSlot], 255u));
    t.alphaSlot = ~0u;
}

PageSet GsVulkanBackend::dirtyTargetPages() const
{
    PageSet s;
    for (const Target *t : m_targets)
        if (t->dirty)
            s |= t->dirtyPages;
    return s;
}

// Upper bound of the texel alpha a GPU-decoded texture can have (for the blend factor check).
uint8_t GsVulkanBackend::estimateMaxAlpha(const GSDrawState &st, const uint32_t *palette, bool overlay, const std::vector<uint16_t> &pages) const
{
    const GSTex0Reg &tex = st.context.tex0;
    switch (tex.psm)
    {
    case GS_PSM_CT24:
        return st.texa.ta0;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return std::max(st.texa.ta0, st.texa.ta1);
    case GS_PSM_CT32:
    {
        if (overlay)
            return 255u;
        uint32_t ma = 0;
        for (uint16_t p : pages)
        {
            const uint8_t *b = m_vram + static_cast<size_t>(p) * 8192u;
            for (uint32_t i = 3; i < 8192u; i += 4u)
                ma = std::max<uint32_t>(ma, b[i]);
        }
        return static_cast<uint8_t>(ma);
    }
    default:
        if (isIndexedPsm(tex.psm) && palette)
        {
            const bool four = tex.psm == GS_PSM_T4 || tex.psm == GS_PSM_T4HL || tex.psm == GS_PSM_T4HH;
            const uint32_t n = four ? 16u : 256u;
            // Indices present in the texture's pages (a superset of the texels): the palette
            // entries they select bound the alpha. Unknown with render-target data: all entries.
            bool used[256] = {};
            if (overlay)
                std::fill(used, used + n, true);
            else
                for (uint16_t p : pages)
                {
                    const uint8_t *b = m_vram + static_cast<size_t>(p) * 8192u;
                    switch (tex.psm)
                    {
                    case GS_PSM_T8: for (uint32_t i = 0; i < 8192u; ++i) used[b[i]] = true; break;
                    case GS_PSM_T4: for (uint32_t i = 0; i < 8192u; ++i) { used[b[i] & 15u] = true; used[b[i] >> 4] = true; } break;
                    case GS_PSM_T8H: for (uint32_t i = 3; i < 8192u; i += 4u) used[b[i]] = true; break;
                    case GS_PSM_T4HL: for (uint32_t i = 3; i < 8192u; i += 4u) used[b[i] & 15u] = true; break;
                    default: for (uint32_t i = 3; i < 8192u; i += 4u) used[b[i] >> 4] = true; break;
                    }
                }
            uint32_t ma = 0;
            for (uint32_t i = 0; i < n; ++i)
                if (used[i])
                    ma = std::max(ma, palette[i] >> 24);
            return static_cast<uint8_t>(ma);
        }
        return 255u;
    }
}

// A texture that is exactly a colour render target (same PSM, base and width): copied on the GPU
// instead of going through GS memory. Only used while the target holds data GS memory lacks.
GsVulkanBackend::Texture *GsVulkanBackend::getAliasTexture(const GSDrawState &st, uint32_t w, uint32_t h, const PageSet &pages, uint32_t &texFlags)
{
    const GSTex0Reg &tex = st.context.tex0;
    if (m_noAlias || (tex.psm != GS_PSM_CT32 && tex.psm != GS_PSM_CT24 && tex.psm != GS_PSM_CT16))
        return nullptr;
    Target *src = nullptr;
    for (Target *t : m_targets)
        if (!t->depth && t->psm == tex.psm && t->fbw == std::max<uint32_t>(tex.tbw, 1u) && (t->fbp << 5) == tex.tbp0)
        {
            src = t;
            break;
        }
    if (!src || !src->dirty)
        return nullptr;
    // Rows no draw ever reached are not render-target data: leave them out (they would pull in
    // whatever lies below the frame in GS memory, e.g. the Z buffer).
    const uint32_t cw = std::min(w, src->img.width), ch = std::min<uint32_t>({h, src->img.height, static_cast<uint32_t>(src->maxRow + 1)});
    // Other targets' newer data inside this target's rows made those rows stale when they were
    // drawn, and uploadStale brings it in; what lies outside the target is not in the alias.
    (void)pages;
    m_why = "texture";
    uploadStale(*src, 0, static_cast<int>(ch) - 1);
    Texture &a = m_alias[src];
    if (!a.img.image)
    {
        createImage(a.img, src->img.width, src->img.height, VK_FORMAT_R8G8B8A8_UINT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT);
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = m_descPool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &m_setLayout;
        m_dt.vkAllocateDescriptorSets(m_device, &dai, &a.set);
        VkDescriptorImageInfo dii{VK_NULL_HANDLE, a.img.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wds.dstSet = a.set;
        wds.descriptorCount = 1;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wds.pImageInfo = &dii;
        m_dt.vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
        a.serial = 0;
    }
    const uint64_t stamp = src->version * 4096u + ch;
    if (a.serial != stamp)
    {
        barrier();
        VkImageCopy ic{};
        ic.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        ic.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        ic.extent = {cw, ch, 1};
        m_dt.vkCmdCopyImage(m_cmd, src->img.image, VK_IMAGE_LAYOUT_GENERAL, a.img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
        barrier();
        a.serial = stamp;
        ++m_statAliasCopies;
    }
    a.lastUse = m_submitSerial;
    a.maxAlpha = tex.psm == GS_PSM_CT32 ? 255u : tex.psm == GS_PSM_CT16 ? std::max(st.texa.ta0, st.texa.ta1) : st.texa.ta0;
    texFlags = tex.psm == GS_PSM_CT16 ? F_TEXA16 : tex.psm == GS_PSM_CT24 ? F_TEXA24 : 0u;
    return &a;
}

GsVulkanBackend::Texture *GsVulkanBackend::getTexture(const GSDrawState &st, uint32_t &decW, uint32_t &decH, uint32_t &texFlags)
{
    texFlags = 0;
    const GSContext &ctx = st.context;
    const GSTex0Reg &tex = ctx.tex0;
    uint32_t w = st.textureWidth, h = st.textureHeight;
    const uint64_t clamp = ctx.clamp;
    const uint32_t wmU = clamp & 3u, wmV = (clamp >> 2) & 3u;
    const uint32_t minU = (clamp >> 4) & 0x3FFu, maxU = (clamp >> 14) & 0x3FFu;
    const uint32_t minV = (clamp >> 24) & 0x3FFu, maxV = (clamp >> 34) & 0x3FFu;
    if (wmU == 2u) w = std::max(w, maxU + 1u);
    if (wmU == 3u) w = std::max(w, (minU | maxU) + 1u);
    if (wmV == 2u) h = std::max(h, maxV + 1u);
    if (wmV == 3u) h = std::max(h, (minV | maxV) + 1u);
    w = std::min<uint32_t>(std::max<uint32_t>(w, 1u), 1024u);
    h = std::min<uint32_t>(std::max<uint32_t>(h, 1u), 1024u);
    decW = w;
    decH = h;

    TexKey key{};
    key.tbp0 = tex.tbp0;
    key.tbw = tex.tbw;
    key.psm = tex.psm;
    key.w = w;
    key.h = h;
    if (tex.psm != GS_PSM_CT32 && !isIndexedPsm(tex.psm))
        key.texa = st.texa.ta0 | (st.texa.ta1 << 8) | (st.texa.aem ? 0x10000u : 0u);
    uint32_t palette[256];
    if (isIndexedPsm(tex.psm))
    {
        const uint64_t pkey = static_cast<uint64_t>(tex.cpsm) | (static_cast<uint64_t>(tex.csm & 1u) << 8) | (static_cast<uint64_t>(tex.csa & 0x1Fu) << 9) |
                              (static_cast<uint64_t>(tex.psm == GS_PSM_T4 || tex.psm == GS_PSM_T4HL || tex.psm == GS_PSM_T4HH) << 14) |
                              (static_cast<uint64_t>(st.texa.ta0) << 16) | (static_cast<uint64_t>(st.texa.ta1) << 24) | (static_cast<uint64_t>(st.texa.aem) << 32);
        if (m_palVersion != m_cpu.ClutVersion())
        {
            m_palCache.clear();
            m_palVersion = m_cpu.ClutVersion();
        }
        auto pit = m_palCache.find(pkey);
        if (pit == m_palCache.end())
        {
            PaletteEntry e;
            const uint32_t n = m_cpu.DecodePalette(st, e.colors.data());
            e.hash = hashWords(e.colors.data(), n) ^ n;
            pit = m_palCache.emplace(pkey, e).first;
        }
        std::memcpy(palette, pit->second.colors.data(), sizeof(palette));
        key.palette = pit->second.hash;
    }

    PageSet pages;
    addRectPages(pages, tex.tbp0, tex.tbw, tex.psm, 0, 0, static_cast<int>(w) - 1, static_cast<int>(h) - 1);
    if (Texture *alias = getAliasTexture(st, w, h, pages, texFlags))
        return alias;
    m_why = "texture";
    if (m_stats)
    {
        static int logged = 0;
        for (Target *t : m_targets)
            if (t->dirty && t->dirtyPages.intersects(pages) && logged < 40)
            {
                ++logged;
                std::fprintf(stderr, "[gs:vk] rt-as-texture: tex tbp=%x tbw=%u psm=%x %ux%u clamp=%llx linear=%d <- target fbp=%x fbw=%u psm=%x dirty=(%d,%d)-(%d,%d)\n",
                             tex.tbp0, tex.tbw, tex.psm, w, h, (unsigned long long)ctx.clamp, st.linearFilter, t->fbp, t->fbw, t->psm, t->dx0, t->dy0, t->dx1, t->dy1);
            }
    }
    // GPU decode (default): texels come from the GPU mirror of GS memory, with any render target
    // holding newer data for these pages written into it first - no GPU->CPU round trip.
    bool overlay = false; // texels include render-target data GS memory does not have yet
    if (m_cpuDecode)
        ensureVramCurrent(pages);
    else
    {
        ensureMirrorCurrent(pages);
        overlay = m_mirrorNewer.intersects(pages);
    }

    auto it = m_textures.find(key);
    if (it != m_textures.end())
    {
        Texture &t = it->second;
        bool valid = true;
        for (uint16_t p : t.pages)
            if (m_cpuDecode ? m_pageSerial[p] > t.serial : m_mirrorGen[p] > t.mirrorGen)
            {
                valid = false;
                break;
            }
        if (!valid && t.rawHash != 0u && !overlay && t.rawHash == pagesHash(t.pages))
        {
            // Pages rewritten with the same bytes (textures streamed in again every frame).
            t.serial = m_serial;
            t.mirrorGen = m_mirrorGenCounter;
            valid = true;
            ++m_statTexRehash;
        }
        if (valid)
        {
            t.lastUse = m_submitSerial;
            ++m_statTexHits;
            resolveMaxAlpha(t);
            return &t;
        }
    }

    ScopeTimer decodeTimer(m_iv.decodeNs);
    ++m_iv.decodes;
    if (m_cpuDecode)
    {
        m_decodeBuf.resize(static_cast<size_t>(w) * h);
        if (!m_cpu.DecodeTexture(st, w, h, isIndexedPsm(tex.psm) ? palette : nullptr, m_decodeBuf.data()))
            return nullptr;
    }
    else if (decodeKind(tex.psm) < 0)
        return nullptr;

    if (it == m_textures.end())
    {
        if (m_textures.size() >= kMaxTextures)
        {
            // Evict the least recently used half (their images are freed after the next submit).
            std::vector<std::pair<uint64_t, TexKey>> order;
            order.reserve(m_textures.size());
            for (auto &kv : m_textures)
                order.emplace_back(kv.second.lastUse, kv.first);
            std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            for (size_t i = 0; i < order.size() / 2; ++i)
            {
                auto e = m_textures.find(order[i].second);
                m_slots[m_cur].defImages.push_back(e->second.img);
                m_slots[m_cur].defSets.push_back(e->second.set);
                m_textures.erase(e);
            }
        }
        it = m_textures.emplace(key, Texture{}).first;
    }
    Texture &t = it->second;
    if (t.img.width != w || t.img.height != h || !t.img.image)
    {
        if (t.img.image)
            m_slots[m_cur].defImages.push_back(t.img);
        t.img = GpuImage{};
        createImage(t.img, w, h, VK_FORMAT_R8G8B8A8_UINT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT);
        if (!t.set)
        {
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = m_descPool;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts = &m_setLayout;
            if (m_dt.vkAllocateDescriptorSets(m_device, &dai, &t.set) != VK_SUCCESS)
                std::fprintf(stderr, "[gs:vk] descriptor set allocation failed\n");
        }
        Texture &tt = it->second;
        VkDescriptorImageInfo dii{VK_NULL_HANDLE, tt.img.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wds.dstSet = tt.set;
        wds.descriptorCount = 1;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wds.pImageInfo = &dii;
        // A set still referenced by recorded commands must not be updated: finish them first.
        if (tt.lastUse == m_submitSerial)
            submitAndWait();
        m_dt.vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
    }
    Texture &tt = it->second;
    tt.pages.clear();
    pages.forEach([&](uint32_t p) { tt.pages.push_back(static_cast<uint16_t>(p)); });
    // Re-writing an image the recorded commands still sample is fine: the write is ordered
    // after them by the barrier.
    if (m_cpuDecode)
    {
        const uint32_t bytes = w * h * 4u;
        const uint32_t off = ringAlloc(bytes, 16u);
        std::memcpy(m_ringPtr + off, m_decodeBuf.data(), bytes);
        barrier();
        VkBufferImageCopy region{};
        region.bufferOffset = off;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {w, h, 1};
        m_dt.vkCmdCopyBufferToImage(m_cmd, m_ring, tt.img.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        barrier();
        uint32_t ma = 0;
        for (uint32_t v : m_decodeBuf)
            ma = std::max(ma, v >> 24);
        tt.maxAlpha = static_cast<uint8_t>(ma);
    }
    else
    {
        gpuDecode(st, tt, pages, palette);
        tt.maxAlpha = estimateMaxAlpha(st, palette, overlay, tt.pages);
        static const bool s_check = envFlag("PS2_GS_VK_CHECKDECODE");
        if (s_check)
        {
            // Debug: compare with the CPU decode of GS memory brought up to date.
            ensureVramCurrent(pages);
            std::vector<uint32_t> ref(static_cast<size_t>(w) * h);
            m_cpu.DecodeTexture(st, w, h, isIndexedPsm(tex.psm) ? palette : nullptr, ref.data());
            if (m_readbackSize < ref.size() * 4u)
            {
                Target dummy;
                (void)dummy;
            }
            if (m_readbackSize >= ref.size() * 4u)
            {
                barrier();
                VkBufferImageCopy region{};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent = {w, h, 1};
                m_dt.vkCmdCopyImageToBuffer(m_cmd, tt.img.image, VK_IMAGE_LAYOUT_GENERAL, m_readback, 1, &region);
                submitAndWait();
                const uint32_t *got = reinterpret_cast<const uint32_t *>(m_readbackPtr);
                size_t bad = 0, first = 0;
                for (size_t i = 0; i < ref.size(); ++i)
                    if (got[i] != ref[i] && bad++ == 0)
                        first = i;
                static int reports = 0;
                if (bad && reports++ < 30)
                    std::fprintf(stderr, "[gs:vk] DECODE MISMATCH tbp=%x tbw=%u psm=%x %ux%u: %zu texels, first (%zu,%zu) gpu %08x cpu %08x overlay=%d\n", tex.tbp0,
                                 tex.tbw, tex.psm, w, h, bad, first % w, first / w, got[first], ref[first], overlay);
            }
        }
    }
    tt.serial = m_serial;
    tt.mirrorGen = m_mirrorGenCounter;
    tt.rawHash = overlay ? 0u : pagesHash(tt.pages); // only meaningful for GS-memory data
    tt.lastUse = m_submitSerial;
    ++m_statTexUploads;
    return &tt;
}

void GsVulkanBackend::flushBatch()
{
    if (!m_batch.active)
        return;
    m_batch.active = false;
    Batch &b = m_batch;
    if (b.verts.empty())
        return;
    ++m_statBatches;
    ++m_iv.batches;
    ScopeTimer flushTimer(m_iv.flushNs);
    const GSDrawState &st = b.state;
    const GSContext &ctx = st.context;
    DrawSetup setup = b.setup;
    const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
    const int tw = static_cast<int>(fbw * 64u), th = static_cast<int>(kTargetHeight);

    // Pixels the batch can touch.
    const int sx0 = std::max<int>(ctx.scissor.x0, 0), sy0 = std::max<int>(ctx.scissor.y0, 0);
    const int sx1 = std::min<int>(ctx.scissor.x1, tw - 1), sy1 = std::min<int>(ctx.scissor.y1, th - 1);
    const int rx0 = std::max(sx0, static_cast<int>(std::floor(b.minX)) - 1);
    const int ry0 = std::max(sy0, static_cast<int>(std::floor(b.minY)) - 1);
    const int rx1 = std::min(sx1, static_cast<int>(std::ceil(b.maxX)) + 1);
    const int ry1 = std::min(sy1, static_cast<int>(std::ceil(b.maxY)) + 1);
    const size_t vertexCount = b.verts.size();
    b.minX = b.minY = 1e30f;
    b.maxX = b.maxY = -1e30f;
    if (rx1 < rx0 || ry1 < ry0 || sx1 < sx0 || sy1 < sy0)
    {
        b.verts.clear();
        return;
    }

    // Texture.
    Texture *tex = nullptr;
    uint32_t decW = 1, decH = 1, texFlags = 0;
    const bool pixelPrims = st.prim.type == GS_PRIM_POINT || st.prim.type == GS_PRIM_LINE || st.prim.type == GS_PRIM_LINESTRIP;
    if (st.prim.tme && !pixelPrims)
    {
        tex = getTexture(st, decW, decH, texFlags);
        if (!tex)
        {
            ++m_statFallback[FB_TEXTURE];
            b.verts.clear();
            return;
        }
    }

    // Upper bound of the blend factor C (As or FIX): above 128 the factor exceeds one.
    uint32_t cBound = 0;
    {
        const uint32_t va = b.maxVA;
        uint32_t as = va;
        if (tex && ctx.tex0.tcc)
        {
            const uint32_t ta = tex->maxAlpha;
            switch (ctx.tex0.tfx & 3u)
            {
            case 0: as = (ta * va) >> 7; break;
            case 2: as = std::min<uint32_t>(255u, ta + va); break;
            default: as = ta; break;
            }
        }
        cBound = (setup.blend.blendWord & 0x100u) ? (setup.blend.blendWord & 0xFFu) : as;
    }
    b.maxVA = 0;
    if (setup.blend.enable && cBound > 128u)
    {
        ++m_statBigFactor[ctx.alpha & 0xFFu];
        if (m_stats && (ctx.alpha & 0xFFu) == 0x44u)
        {
            static int n = 0;
            if (n++ < 12)
                std::fprintf(stderr, "[gs:vk] big factor 0x44: tex psm=%x tbp=%x maxA=%u tcc=%d tfx=%d vA=%u cBound=%u\n", ctx.tex0.psm, ctx.tex0.tbp0,
                             tex ? tex->maxAlpha : 0u, ctx.tex0.tcc, ctx.tex0.tfx, b.maxVA, cBound);
        }
        if ((setup.blend.flags & F_BLEND_SCALE) && !m_noBigDst)
        {
            // A blend factor above one cannot be done with fixed-function blending: blend in the
            // shader from a copy of the target instead (exact GS arithmetic).
            setup.dstRead = true;
            setup.blend.enable = false;
            setup.blend.flags = F_DSTREAD | F_BLEND_DST;
            setup.fbWriteMask = ctx.frame.psm == GS_PSM_CT24 ? 0x7u : 0xFu;
        }
    }

    // Targets.
    bool writesColor = false, writesDepth = false;
    for (int i = 0; i < setup.passCount; ++i)
    {
        const uint32_t cm = ((setup.passes[i].rgb ? 7u : 0u) | (setup.passes[i].alpha ? 8u : 0u)) & setup.fbWriteMask;
        writesColor = writesColor || cm != 0u;
        writesDepth = writesDepth || setup.passes[i].depth;
    }
    Target &color = getTarget(ctx.frame.fbp, fbw, ctx.frame.psm);
    PageSet colorPages;
    if (writesColor)
        addRectPages(colorPages, ctx.frame.fbp << 5, fbw, ctx.frame.psm, rx0, ry0, rx1, ry1);
    prepareDrawTarget(color, colorPages, ry0, ry1);
    Target *depth = nullptr;
    PageSet depthPages;
    if (setup.needDepth)
    {
        depth = &getTarget(ctx.zbuf.zbp, fbw, ctx.zbuf.psm);
        if (writesDepth)
            addRectPages(depthPages, ctx.zbuf.zbp << 5, fbw, ctx.zbuf.psm, rx0, ry0, rx1, ry1);
        prepareDrawTarget(*depth, depthPages, ry0, ry1);
    }
    // Preparing the depth target may have downloaded the colour target (overlap): re-upload.
    uploadStale(color, ry0, ry1);

    // Destination copy for draws that read the target.
    if (setup.dstRead)
    {
        ensureDstImage(color.img.width);
        barrier();
        VkImageCopy ic{};
        ic.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        ic.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        ic.srcOffset = {rx0, ry0, 0};
        ic.dstOffset = {rx0, ry0, 0};
        ic.extent = {static_cast<uint32_t>(rx1 - rx0 + 1), static_cast<uint32_t>(ry1 - ry0 + 1), 1};
        m_dt.vkCmdCopyImage(m_cmd, color.img.image, VK_IMAGE_LAYOUT_GENERAL, m_dstImg.image, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
        barrier();
        ++m_statDstCopies;
    }

    // Vertices.
    const uint32_t vbytes = static_cast<uint32_t>(vertexCount * sizeof(GpuVertex));
    const uint32_t voff = ringAlloc(vbytes, 32u);
    std::memcpy(m_ringPtr + voff, b.verts.data(), vbytes);
    b.verts.clear();

    // Rendering.
    if (!m_rendering || m_curColor != &color || m_curDepth != depth)
    {
        endRendering();
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ca.imageView = color.img.view;
        ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (depth)
        {
            da.imageView = depth->img.view;
            da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {color.img.width, color.img.height}};
        if (depth)
        {
            ri.renderArea.extent.width = std::min(color.img.width, depth->img.width);
            ri.renderArea.extent.height = std::min(color.img.height, depth->img.height);
        }
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &ca;
        ri.pDepthAttachment = depth ? &da : nullptr;
        m_dt.vkCmdBeginRendering(m_cmd, &ri);
        m_rendering = true;
        m_curColor = &color;
        m_curDepth = depth;
        m_curPipeline = VK_NULL_HANDLE;
    }
    const VkViewport vp{0.0f, 0.0f, 4096.0f, 4096.0f, 0.0f, 1.0f};
    m_dt.vkCmdSetViewport(m_cmd, 0, 1, &vp);
    const VkRect2D sc{{sx0, sy0}, {static_cast<uint32_t>(sx1 - sx0 + 1), static_cast<uint32_t>(sy1 - sy0 + 1)}};
    m_dt.vkCmdSetScissor(m_cmd, 0, 1, &sc);
    const VkDeviceSize vo = voff;
    m_dt.vkCmdBindVertexBuffers(m_cmd, 0, 1, &m_ring, &vo);
    const VkDescriptorSet sets[2] = {tex ? tex->set : m_dummySet, m_dstSet};
    m_dt.vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeLayout, 0, 2, sets, 0, nullptr);

    PushConsts pc{};
    uint32_t flags = setup.blend.flags | texFlags;
    if (tex)
    {
        flags |= F_TME;
        if (st.prim.fst)
            flags |= st.prim.type == GS_PRIM_SPRITE ? F_FST_ROUND : F_FST_TRUNC;
        if (st.linearFilter)
            flags |= F_LINEAR;
        if (ctx.tex0.tcc)
            flags |= F_TCC;
        flags |= static_cast<uint32_t>(ctx.tex0.tfx & 3u) << 5;
    }
    if (st.prim.fge)
        flags |= F_FGE;
    if (st.prim.iip && st.prim.type != GS_PRIM_SPRITE && !pixelPrims)
        flags |= F_IIP;
    if (ctx.frame.psm == GS_PSM_CT16)
        flags |= F_CT16;
    if (depth && ctx.zbuf.psm != GS_PSM_Z32)
        flags |= F_ZROUND;
    if (st.prim.type == GS_PRIM_SPRITE || pixelPrims)
        flags |= F_ZFLAT;
    // Fixed-function blending rounds to nearest where the GS truncates: bias the source term by
    // just under half a step (towards a smaller result) so the blend truncates too.
    if ((flags & F_BLEND_SCALE) && !m_noBias)
        flags |= setup.blend.op == 2u ? F_BIAS_UP : F_BIAS_DOWN;
    pc.wrap = static_cast<uint32_t>(ctx.clamp & 0xFu);
    pc.regionU = static_cast<uint32_t>((ctx.clamp >> 4) & 0x3FFu) | (static_cast<uint32_t>((ctx.clamp >> 14) & 0x3FFu) << 16);
    pc.regionV = static_cast<uint32_t>((ctx.clamp >> 24) & 0x3FFu) | (static_cast<uint32_t>((ctx.clamp >> 34) & 0x3FFu) << 16);
    pc.texW = st.textureWidth;
    pc.texH = st.textureHeight;
    pc.fogCol = st.fogR | (st.fogG << 8) | (st.fogB << 16);
    pc.kS = setup.blend.kS;
    pc.dS = setup.blend.dS;
    pc.blend = setup.blend.blendWord;
    pc.kD = setup.blend.kD;
    pc.dD = setup.blend.dD;
    pc.fbmsk = ctx.frame.fbmsk;
    pc.texa = st.texa.ta0 | (static_cast<uint32_t>(st.texa.ta1) << 8) | (st.texa.aem ? 0x10000u : 0u);
    const uint32_t atstAref = static_cast<uint32_t>((ctx.test >> 1) & 7u) | (static_cast<uint32_t>((ctx.test >> 4) & 0xFFu) << 8);
    const bool fba = (ctx.fba & 1u) != 0u && ctx.frame.psm != GS_PSM_CT24;

    static const VkCompareOp zops[4] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_GREATER};
    auto draw = [&](uint32_t key, uint32_t drawFlags, uint32_t atest, bool depthWrite, VkCompareOp zop)
    {
        VkPipeline pipe = getPipeline(key);
        if (!pipe)
            return;
        if (pipe != m_curPipeline)
        {
            m_dt.vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            m_curPipeline = pipe;
        }
        pc.flags = drawFlags;
        pc.atest = atest;
        m_dt.vkCmdPushConstants(m_cmd, m_pipeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConsts), &pc);
        m_dt.vkCmdSetDepthTestEnable(m_cmd, depth ? VK_TRUE : VK_FALSE);
        m_dt.vkCmdSetDepthWriteEnable(m_cmd, (depth && depthWrite) ? VK_TRUE : VK_FALSE);
        m_dt.vkCmdSetDepthCompareOp(m_cmd, depth ? zop : VK_COMPARE_OP_ALWAYS);
        m_dt.vkCmdDraw(m_cmd, static_cast<uint32_t>(vertexCount), 1, 0, 0);
        ++m_statDraws;
    };
    const bool factorInAlpha = setup.blend.enable && !m_dualSrc && setup.blend.dst >= 2u;
    // (Cd - Cs) * C with C above 128: blend factors are clamped to one, so those pixels take
    // (Cd - Cs) first and then Cd * C/128 as Cd + Cd * (C/128 - 1).
    const bool split = setup.blend.enable && cBound > 128u && setup.blend.kS == -1 && setup.blend.dS == 0 && setup.blend.dst == 2u && setup.blend.op == 2u;
    for (int i = 0; i < setup.passCount; ++i)
    {
        const PassSetup &p = setup.passes[i];
        const uint32_t cm = ((p.rgb ? 7u : 0u) | (p.alpha ? 8u : 0u)) & setup.fbWriteMask;
        if (cm == 0u && !p.depth)
            continue;
        const uint32_t atest = atstAref | (p.ate ? 0x10000u : 0u) | (p.invert ? 0x20000u : 0u);
        const uint32_t passFlags = flags | ((fba && (cm & 8u)) ? F_FBA : 0u);
        const VkCompareOp zop = zops[setup.ztest & 3u];
        const uint32_t dkey = depth ? 0x200u : 0u;
        const uint32_t base = (setup.blend.enable ? 1u : 0u) | (static_cast<uint32_t>(setup.blend.op) << 1) |
                              (static_cast<uint32_t>(setup.blend.dst) << 3) | dkey;
        if (setup.blend.adHalf)
        {
            // Cs * Ad/128 + Cd = Cd + 2 * (Cs/256 * Ad/255 * 255): two additive draws (saturating
            // adds commute), then the alpha channel on its own.
            if (cm & 7u)
            {
                const uint32_t k = 1u | (1u << 3) | ((cm & 7u) << 5) | dkey | 0x1000u;
                draw(k, passFlags | F_AD_HALF, atest, p.depth, zop);
                draw(k, passFlags | F_AD_HALF, atest, false, p.depth ? VK_COMPARE_OP_EQUAL : zop);
            }
            if (cm & 8u)
                draw(dkey | (8u << 5), passFlags & ~(F_BLEND_EXACT | F_BLEND_SCALE | F_BLEND_INT), atest, (cm & 7u) ? false : p.depth,
                     ((cm & 7u) && p.depth) ? VK_COMPARE_OP_EQUAL : zop);
            else if (!(cm & 7u) && p.depth)
                draw(dkey, passFlags, atest, true, zop);
            continue;
        }
        const bool rgbOnly = factorInAlpha || split;
        if (!rgbOnly || (cm & 7u) == 0u)
        {
            draw(base | (cm << 5), passFlags, atest, p.depth, zop);
            continue;
        }
        const uint32_t rgbMask = (cm & 7u) << 5;
        const uint32_t lowFlags = passFlags | (split ? F_SPLIT_LOW : 0u);
        if (factorInAlpha)
            draw(base | rgbMask | 0x400u, lowFlags | F_ALPHA_FACTOR, atest, p.depth, zop);
        else
            draw(base | rgbMask, lowFlags, atest, p.depth, zop);
        if (split)
        {
            const int32_t kS = pc.kS, dS = pc.dS;
            pc.kS = 0;
            pc.dS = 1; // source = Cs
            draw(1u | (2u << 1) | (1u << 3) | rgbMask | dkey, passFlags | F_SPLIT_HIGH, atest, p.depth, zop);
            pc.kS = kS;
            pc.dS = dS;
            draw(1u | (0u << 1) | (1u << 3) | rgbMask | dkey | 0x800u, (passFlags | F_SPLIT_HIGH | F_OUT_FM1) & ~F_BLEND_SCALE, atest, false,
                 p.depth ? VK_COMPARE_OP_EQUAL : zop);
        }
        if (cm & 8u)
            draw(dkey | (8u << 5), passFlags & ~(F_BLEND_EXACT | F_BLEND_SCALE | F_BLEND_INT), atest, false, p.depth ? VK_COMPARE_OP_EQUAL : zop);
    }
    if (tex)
        tex->lastUse = m_submitSerial;

    auto grow = [&](Target &t, const PageSet &pages)
    {
        if (!t.dirty)
        {
            t.dirty = true;
            t.dx0 = rx0;
            t.dy0 = ry0;
            t.dx1 = rx1;
            t.dy1 = ry1;
        }
        else
        {
            t.dx0 = std::min(t.dx0, rx0);
            t.dy0 = std::min(t.dy0, ry0);
            t.dx1 = std::max(t.dx1, rx1);
            t.dy1 = std::max(t.dy1, ry1);
        }
        t.dirtyPages |= pages;
        t.maxRow = std::max(t.maxRow, ry1);
    };
    if (writesColor)
    {
        grow(color, colorPages);
        ++color.version;
    }
    if (depth && writesDepth)
    {
        grow(*depth, depthPages);
        ++depth->version;
    }
}

// ------------------------------------------------------------------------------------------
// GSRasterBackend
// ------------------------------------------------------------------------------------------

void GsVulkanBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    flushBatch();
    dropAll();
    m_cpu.Initialize(vram, vramSize);
    m_vram = vram;
    m_vramSize = vramSize;
    m_pageSerial.fill(0);
    m_serial = 1;
    m_mirrorSerial.fill(~0ull);
    m_mirrorNewer.reset();
}

void GsVulkanBackend::Reset()
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    flushBatch();
    downloadAll(true);
    m_cpu.Reset();
}

void GsVulkanBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!isIndexedPsm(tex0.psm) || tex0.cld == 0u || tex0.cld >= 6u)
        return; // no CLUT load (the CPU backend ignores these too)
    ScopeTimer timer(m_iv.clutNs);
    flushBatch();
    PageSet pages;
    // CSM1: a 16x16 (or 8x2) block-ordered CLUT occupies at most four blocks from CBP.
    pages.set((tex0.cbp >> 5) & 511u);
    pages.set(((tex0.cbp + 3u) >> 5) & 511u);
    if (tex0.csm)
        addRectPages(pages, tex0.cbp, texclut.cbw, tex0.cpsm, texclut.cou * 16, texclut.cov, texclut.cou * 16 + 255, texclut.cov + 1);
    m_why = "clut";
    if (m_stats)
    {
        static int logged = 0;
        for (Target *t : m_targets)
            if (t->dirty && t->dirtyPages.intersects(pages) && logged < 30)
            {
                ++logged;
                std::fprintf(stderr, "[gs:vk] clut cbp=%x cpsm=%x csm=%u csa=%u psm=%x <- target fbp=%x fbw=%u psm=%x dirty=(%d,%d)-(%d,%d)\n", tex0.cbp, tex0.cpsm, tex0.csm,
                             tex0.csa, tex0.psm, t->fbp, t->fbw, t->psm, t->dx0, t->dy0, t->dx1, t->dy1);
            }
    }
    ensureVramCurrent(pages);
    m_cpu.LoadClut(tex0, texclut);
}

void GsVulkanBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    ScopeTimer timer(m_iv.xferNs);
    flushBatch();
    const auto &bb = command.bitbltbuf;
    const auto &pos = command.trxpos;
    const auto &reg = command.trxreg;
    PageSet src, dst;
    if (reg.rrw && reg.rrh)
    {
        addRectPages(src, bb.sbp, bb.sbw, bb.spsm, pos.ssax, pos.ssay, pos.ssax + reg.rrw - 1, pos.ssay + reg.rrh - 1);
        addRectPages(dst, bb.dbp, bb.dbw, bb.dpsm, pos.dsax, pos.dsay, pos.dsax + reg.rrw - 1, pos.dsay + reg.rrh - 1);
    }
    m_uploadPages.reset();
    m_why = command.direction == 0u ? "xfer-up" : command.direction == 1u ? "xfer-down" : "xfer-copy";
    switch (command.direction)
    {
    case 0u:
        ensureVramCurrent(dst);
        m_cpu.BeginTransfer(command);
        m_uploadPages = dst;
        break;
    case 1u:
        ensureVramCurrent(src);
        m_cpu.BeginTransfer(command);
        break;
    case 2u:
        ensureVramCurrent(src | dst);
        m_cpu.BeginTransfer(command);
        markCpuWrite(dst);
        break;
    default:
        m_cpu.BeginTransfer(command);
        break;
    }
}

void GsVulkanBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    ScopeTimer timer(m_iv.xferNs);
    flushBatch();
    ensureVramCurrent(m_uploadPages);
    m_cpu.UploadImage(data, sizeBytes);
    markCpuWrite(m_uploadPages);
}

void GsVulkanBackend::Flush()
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    flushBatch();
}

void GsVulkanBackend::Sync(GSSyncReason reason)
{
    if (reason == GSSyncReason::Presentation)
        return; // Present() takes what it needs
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    flushBatch();
    m_why = "sync";
    downloadAll(true);
}

void GsVulkanBackend::QueuePresentSnapshot(const GSPresentationRequest &request)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    flushBatch();
    ScopeTimer flipTimer(m_iv.flipNs);
    m_why = "flip";
    ++m_statFlips;
    if (m_stats && (m_statFlips % 300u) == 0u)
        printStats();
    const PageSet pages = displayPages(request);
    DispParams dp;
    const bool gpuFlip = m_gpuPresent && gpuMem() && buildDisplayParams(request, dp);
    if (!gpuFlip && m_gpuPresent)
    {
        std::lock_guard<std::mutex> lock(m_gpuFrameMutex);
        m_pubGpu = false; // this picture goes through the CPU
    }
    if (gpuFlip && !m_checkPresent)
    {
        // Build the picture on the GPU; the host shows it once this submission has run.
        ensureMirrorCurrent(pages);
        const int image = pickPresentImage();
        recordDisplay(dp, image);
        Slot &sl = m_slots[m_cur];
        sl.gpuImage = image;
        sl.gpuW = dp.outW;
        sl.gpuH = dp.outH;
        ++m_statGpuFlips;
        submitAsync();
        return;
    }
    if (gpuMem() && !m_syncPresent)
    {
        // Copy the display pages out of the mirror at this point of the command stream, submit
        // without waiting, and publish them when the GPU is done (one frame later): the GPU
        // renders this frame while the game prepares the next.
        ensureMirrorCurrent(pages);
        Slot &sl = m_slots[m_cur];
        std::vector<VkBufferCopy> copies;
        sl.presentPages.clear();
        for (uint32_t p = 0; p < 512u; ++p)
            if (pages.test(p))
            {
                copies.push_back({static_cast<VkDeviceSize>(p) * 8192u, static_cast<VkDeviceSize>(p) * 8192u, 8192u});
                sl.presentPages.push_back(static_cast<uint16_t>(p));
            }
        barrier();
        if (!copies.empty())
            m_dt.vkCmdCopyBuffer(m_cmd, m_vramBuf, sl.presentBuf, static_cast<uint32_t>(copies.size()), copies.data());
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        sl.presentReq = request;
        sl.hasPresent = true;
        if (gpuFlip)
        {
            // PS2_GS_VK_CHECKPRESENT: the display pass too, read back and compared with the CPU
            // picture of the same snapshot when this slot completes.
            const int image = pickPresentImage();
            recordDisplay(dp, image);
            Slot &cs = m_slots[m_cur];
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {kPresentW, kPresentH, 1};
            m_dt.vkCmdCopyImageToBuffer(m_cmd, m_presentImg[image].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, cs.checkBuf, 1, &region);
            m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            cs.hasCheck = true;
            cs.gpuW = dp.outW;
            cs.gpuH = dp.outH;
            if (m_hostPresent)
                cs.gpuImage = image;
            ++m_statGpuFlips;
        }
        submitAsync();
        return;
    }
    ensureVramCurrent(pages);
    std::lock_guard<std::mutex> plock(m_presentMutex);
    m_snapLatest.assign(m_vram, m_vram + m_vramSize);
    m_snapRequest = request;
    m_haveSnap = true;
    m_lastFlipNs.store(nowNs(), std::memory_order_relaxed);
}

// The CPU presenter's choice of picture (GSCpuBackend::PresentFromLocalMemory) as display pass
// parameters. False when that path would show nothing: the flip then goes through the CPU.
bool GsVulkanBackend::buildDisplayParams(const GSPresentationRequest &request, DispParams &dp) const
{
    dp = DispParams{};
    auto kindOf = [](uint32_t psm) -> int
    {
        switch (psm)
        {
        case GS_PSM_CT32: return 0;
        case GS_PSM_CT24: return 1;
        case GS_PSM_CT16: return 2;
        case GS_PSM_CT16S: return 3;
        default: return -1;
        }
    };
    struct Circuit
    {
        bool valid = false;
        uint32_t fbp = 0, fbw = 0, psm = 0, ox = 0, oy = 0, w = 0, h = 0;
    } crt[2];
    const uint64_t pmode = request.pmode;
    const uint64_t dispfb[2] = {request.dispfb1, request.dispfb2};
    const uint64_t display[2] = {request.display1, request.display2};
    for (int c = 0; c < 2; ++c)
    {
        Circuit &k = crt[c];
        k.fbp = static_cast<uint32_t>(dispfb[c] & 0x1FFu);
        k.fbw = static_cast<uint32_t>((dispfb[c] >> 9) & 0x3Fu);
        k.psm = static_cast<uint32_t>((dispfb[c] >> 15) & 0x1Fu);
        k.ox = static_cast<uint32_t>((dispfb[c] >> 32) & 0x7FFu);
        k.oy = static_cast<uint32_t>((dispfb[c] >> 43) & 0x7FFu);
        const uint32_t dw = static_cast<uint32_t>((display[c] >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display[c] >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display[c] >> 23) & 0x0Fu);
        k.w = (dw + 1u) / (magh + 1u);
        k.h = dh + 1u;
        if (k.w < 64u || k.h < 64u)
        {
            k.w = 640u;
            k.h = 448u;
        }
        k.w = std::min(k.w, kPresentW);
        k.h = std::min(k.h, kPresentH);
        k.valid = ((pmode >> c) & 1u) != 0u && (k.fbw != 0u || dw != 0u || dh != 0u || magh != 0u);
    }
    if (!crt[0].valid && !crt[1].valid)
        return false;

    // Circuit c's display buffer (or the preferred source), and its fallbacks.
    auto setCircuit = [&](int c, bool allowPreferred) -> bool
    {
        const Circuit &k = crt[c];
        DispSrc *s = &dp.src[c * 3];
        bool usedPreferred = false;
        const GSFrameReg &pref = request.preferredSource;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == k.fbp && (pref.fbw != 0u || pref.fbp != k.fbp) &&
            kindOf(pref.psm) >= 0)
        {
            // CopyFrameToHostRgba(preferredSource, ..., frameBaseIsPages = false): its fbp is a block address.
            s[0] = {1u, pref.fbp, pref.fbw ? pref.fbw : 10u, static_cast<uint32_t>(kindOf(pref.psm)), 0u, 0u, k.w, k.h};
            usedPreferred = true;
        }
        if (!usedPreferred)
        {
            if (kindOf(k.psm) < 0)
                return false;
            s[0] = {1u, k.fbp << 5, k.fbw ? k.fbw : 10u, static_cast<uint32_t>(kindOf(k.psm)), k.ox, k.oy, k.w, k.h};
        }
        if (!usedPreferred && k.fbp == 0u)
        {
            dp.fallback |= 1u << c;
            int n = 1;
            for (const GSFrameReg &f : request.contextFrames)
            {
                if ((f.fbp == k.fbp && f.fbw == k.fbw && f.psm == k.psm) || kindOf(f.psm) < 0)
                    continue;
                s[n++] = {1u, f.fbp << 5, f.fbw ? f.fbw : 10u, static_cast<uint32_t>(kindOf(f.psm)), 0u, 0u, k.w, k.h};
            }
        }
        return true;
    };
    bool merged = false;
    if (crt[0].valid && crt[1].valid)
    {
        if (setCircuit(0, false) && setCircuit(1, false))
        {
            merged = true;
            dp.circuits = 3u;
            dp.outW = std::max(crt[0].w, crt[1].w);
            dp.outH = std::max(crt[0].h, crt[1].h);
        }
        else
            dp = DispParams{};
    }
    if (!merged)
    {
        const int c = crt[0].valid ? 0 : 1;
        if (!setCircuit(c, true))
            return false;
        dp.circuits = 1u << c;
        dp.outW = crt[c].w;
        dp.outH = crt[c].h;
    }
    dp.mmod = static_cast<uint32_t>((pmode >> 5) & 1u);
    dp.amod = static_cast<uint32_t>((pmode >> 6) & 1u);
    dp.slbg = static_cast<uint32_t>((pmode >> 7) & 1u);
    dp.alp = static_cast<uint32_t>((pmode >> 8) & 0xFFu);
    dp.bgcolor = static_cast<uint32_t>(request.bgcolor & 0xFFFFFFu);
    static const bool s_bob = []
    {
        const char *v = std::getenv("PS2_FIELD_PRESENT");
        return v && (v[0] == 'b' || v[0] == 'B');
    }();
    const bool interlaced = (request.smode2 & 1u) != 0u, frameMode = ((request.smode2 >> 1) & 1u) != 0u;
    dp.bob = (s_bob && interlaced && !frameMode) ? 1u : 0u;
    dp.oddField = static_cast<uint32_t>(request.vsyncTick & 1u);
    dp.tables[0] = m_lutOffset[0];
    dp.tables[1] = m_lutOffset[1];
    dp.tables[2] = m_lutOffset[2];
    return true;
}

// A display image the host is not reading and will not pick up (not the published one).
int GsVulkanBackend::pickPresentImage()
{
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        uint64_t released = 0;
        if (m_releaseSem)
            m_dt.vkGetSemaphoreCounterValue(m_device, m_releaseSem, &released);
        uint64_t oldest = ~0ull;
        {
            std::lock_guard<std::mutex> lock(m_gpuFrameMutex);
            for (uint32_t k = 0; k < kPresentImages; ++k)
            {
                const uint32_t i = (m_presentNext + k) % kPresentImages;
                if (static_cast<int>(i) == m_pubImage)
                    continue;
                if (m_presentLastUse[i] <= released)
                {
                    m_presentNext = (i + 1u) % kPresentImages;
                    return static_cast<int>(i);
                }
                oldest = std::min(oldest, m_presentLastUse[i]);
            }
        }
        // All being shown (should not happen with four): wait for the oldest, briefly.
        ScopeTimer timer(m_iv.waitNs);
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &m_releaseSem;
        wi.pValues = &oldest;
        m_dt.vkWaitSemaphores(m_device, &wi, 100000000ull);
    }
    const uint32_t i = m_presentNext;
    m_presentNext = (i + 1u) % kPresentImages;
    return static_cast<int>(i == static_cast<uint32_t>(m_pubImage) ? (i + 1u) % kPresentImages : i);
}

void GsVulkanBackend::recordDisplay(const DispParams &dp, int image)
{
    reserve(static_cast<uint32_t>(sizeof(DispParams)) + static_cast<uint32_t>(m_ssboAlign), 1u);
    const uint32_t off = ringAlloc(static_cast<uint32_t>(sizeof(DispParams)), static_cast<uint32_t>(m_ssboAlign));
    std::memcpy(m_ringPtr + off, &dp, sizeof(DispParams));
    VkDescriptorSet set = computeSet();
    GpuImage &img = m_presentImg[image];
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, img.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo pb{m_ring, off, sizeof(DispParams)}, cb{m_dispCount, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[3]{};
    for (int i = 0; i < 3; ++i)
    {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set;
        w[i].descriptorCount = 1;
    }
    w[0].dstBinding = 3;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w[0].pImageInfo = &ii;
    w[1].dstBinding = 4;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &pb;
    w[2].dstBinding = 5;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[2].pBufferInfo = &cb;
    m_dt.vkUpdateDescriptorSets(m_device, 3, w, 0, nullptr);

    barrier();
    m_dt.vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_displayPipe);
    m_dt.vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compPipeLayout, 0, 1, &set, 0, nullptr);
    uint32_t pass = 0;
    if (dp.fallback)
    {
        m_dt.vkCmdFillBuffer(m_cmd, m_dispCount, 0, VK_WHOLE_SIZE, 0u);
        barrier();
        m_dt.vkCmdPushConstants(m_cmd, m_compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pass), &pass);
        m_dt.vkCmdDispatch(m_cmd, kPresentW / 8u, kPresentH / 8u, 1);
        barrier();
    }
    VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ib.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; // whatever the last picture left is not needed
    ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = img.image;
    ib.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    pass = 1;
    m_dt.vkCmdPushConstants(m_cmd, m_compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pass), &pass);
    m_dt.vkCmdDispatch(m_cmd, kPresentW / 8u, kPresentH / 8u, 1);
    // Ready for the host's blit (or the check readback).
    ib.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    m_dt.vkCmdPipelineBarrier(m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &ib);
    m_curPipeline = VK_NULL_HANDLE;
}

// PS2_GS_VK_CHECKPRESENT: compares the display pass's picture with the CPU presenter's.
void GsVulkanBackend::checkPresent(Slot &sl, const std::vector<uint8_t> &snap)
{
    sl.hasCheck = false;
    if (!sl.checkCoherent)
    {
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = sl.checkMem;
        r.size = VK_WHOLE_SIZE;
        m_dt.vkInvalidateMappedMemoryRanges(m_device, 1, &r);
    }
    std::vector<uint8_t> copy = snap;
    m_presenter.Initialize(copy.data(), static_cast<uint32_t>(copy.size()));
    const PresentationFrame ref = m_presenter.Present(sl.presentReq);
    ++m_statChecks;
    uint32_t bad = 0, firstX = 0, firstY = 0;
    const bool sizeOk = ref.width == sl.gpuW && ref.height == sl.gpuH;
    if (ref && sizeOk)
        for (uint32_t y = 0; y < ref.height; ++y)
            for (uint32_t x = 0; x < ref.width; ++x)
            {
                const uint8_t *a = ref.pixels.data() + (static_cast<size_t>(y) * 640u + x) * 4u;
                const uint8_t *b = sl.checkPtr + (static_cast<size_t>(y) * kPresentW + x) * 4u;
                if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2])
                {
                    if (bad++ == 0)
                    {
                        firstX = x;
                        firstY = y;
                    }
                }
            }
    if (!ref || !sizeOk || bad)
    {
        ++m_statCheckBad;
        if (m_statCheckBad <= 20u)
            std::fprintf(stderr, "[gs:vk] present check: cpu %ux%u gpu %ux%u, %u pixels differ (first %u,%u)\n", ref.width, ref.height, sl.gpuW,
                         sl.gpuH, bad, firstX, firstY);
    }
}

bool GsVulkanBackend::provideFrame(void *user, HostGpuFrame &out, uint64_t releaseValue)
{
#if defined(PS2X_HOST_SDL3)
    GsVulkanBackend *self = static_cast<GsVulkanBackend *>(user);
    std::lock_guard<std::mutex> lock(self->m_gpuFrameMutex);
    // Follow the runtime's choice for this host frame (it skipped or made the CPU picture).
    if (!self->m_hostUseGpu || self->m_pubImage < 0)
        return false;
    const int i = self->m_pubImage;
    out.image = self->m_presentImg[i].image;
    out.imageWidth = kPresentW;
    out.imageHeight = kPresentH;
    out.width = self->m_pubW;
    out.height = self->m_pubH;
    out.ready = self->m_readySem;
    out.readyValue = self->m_pubValue;
    self->m_presentLastUse[i] = releaseValue;
    return true;
#else
    (void)user;
    (void)out;
    (void)releaseValue;
    return false;
#endif
}

bool GsVulkanBackend::PresentsOnGpu() const
{
    if (!m_hostPresent)
        return false;
    // While the game flips, the picture of its last flip; when it stops flipping (or a flip
    // needs the CPU path), Present() snapshots on the CPU as before.
    std::lock_guard<std::mutex> lock(m_gpuFrameMutex);
    m_hostUseGpu = m_pubGpu && m_pubImage >= 0 && nowNs() - m_pubNs <= 250000000ull;
    return m_hostUseGpu;
}

PresentationFrame GsVulkanBackend::Present(const GSPresentationRequest &request)
{
    GSPresentationRequest req = request;
    const uint64_t lastFlip = m_lastFlipNs.load(std::memory_order_relaxed);
    bool have = false;
    if (lastFlip != 0u && nowNs() - lastFlip < 250000000ull)
    {
        std::lock_guard<std::mutex> plock(m_presentMutex);
        if (m_haveSnap)
        {
            m_presentVram = m_snapLatest;
            req = m_snapRequest;
            have = true;
        }
    }
    if (!have)
    {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        if (!m_vram)
            return {};
        flushBatch();
        ensureVramCurrent(displayPages(req));
        m_presentVram.assign(m_vram, m_vram + m_vramSize);
    }
    m_presenter.Initialize(m_presentVram.data(), static_cast<uint32_t>(m_presentVram.size()));
    return m_presenter.Present(req);
}

bool GsVulkanBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    flushBatch();
    PageSet pages;
    addRectPages(pages, context.frame.fbp << 5, std::max<uint32_t>(context.frame.fbw, 1u), context.frame.psm,
                 context.scissor.x0, context.scissor.y0, context.scissor.x1, context.scissor.y1);
    ensureVramCurrent(pages);
    const bool done = m_cpu.ClearFramebuffer(context, rgba);
    if (done)
        markCpuWrite(pages);
    return done;
}

uint32_t GsVulkanBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_cpu.ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GsVulkanBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    auto *self = const_cast<GsVulkanBackend *>(this);
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    self->flushBatch();
    PageSet pages;
    addRectPages(pages, base, bw, static_cast<uint8_t>(psm), static_cast<int>(x), static_cast<int>(y), static_cast<int>(x), static_cast<int>(y));
    self->ensureVramCurrent(pages);
    return m_cpu.ReadVram(psm, base, bw, x, y);
}

void GsVulkanBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    flushBatch();
    PageSet pages;
    addRectPages(pages, base, bw, static_cast<uint8_t>(psm), static_cast<int>(x), static_cast<int>(y), static_cast<int>(x), static_cast<int>(y));
    ensureVramCurrent(pages);
    m_cpu.WriteVram(psm, base, bw, x, y, value);
    markCpuWrite(pages);
}

void GsVulkanBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    auto *self = const_cast<GsVulkanBackend *>(this);
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    self->flushBatch();
    self->downloadAll(true);
    m_cpu.SnapshotVram(out);
}

GSTransferSnapshot GsVulkanBackend::GetTransferSnapshot() const
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_cpu.GetTransferSnapshot();
}

void GsVulkanBackend::printStats()
{
    {
        const double f = static_cast<double>(std::max<uint64_t>(m_statFlips - m_ivFlips, 1u));
        const auto ms = [&](uint64_t ns) { return static_cast<double>(ns) / 1e6 / f; };
        const auto pf = [&](uint64_t n) { return static_cast<double>(n) / f; };
        std::fprintf(stderr,
                     "[gs:vk] per frame: submit %.2f ms (flush %.2f), GPU waits %.1fx %.2f ms, downloads %.1fx %.2f ms, texture decodes %.1fx %.2f ms, "
                     "target uploads %.1fx %.2f ms, transfers %.2f ms, CLUT %.2f ms, flip %.2f ms | %.0f prims, %.0f batches\n",
                     ms(m_iv.submitNs), ms(m_iv.flushNs), pf(m_iv.waits), ms(m_iv.waitNs), pf(m_iv.downloads), ms(m_iv.downloadNs), pf(m_iv.decodes),
                     ms(m_iv.decodeNs), pf(m_iv.uploads), ms(m_iv.uploadNs), ms(m_iv.xferNs), ms(m_iv.clutNs), ms(m_iv.flipNs), pf(m_iv.prims), pf(m_iv.batches));
        m_iv = Interval{};
        m_ivFlips = m_statFlips;
    }
    std::fprintf(stderr,
                 "[gs:vk] mirror pages=%llu overlays=%llu alias copies=%llu dst copies=%llu prims=%llu batches=%llu draws=%llu submits=%llu downloads=%llu (%llu px) uploads=%llu (%llu rows) tex uploads=%llu hits=%llu (rehash %llu) targets=%zu textures=%zu\n",
                 (unsigned long long)m_statMirrorPages, (unsigned long long)m_statOverlays, (unsigned long long)m_statAliasCopies, (unsigned long long)m_statDstCopies, (unsigned long long)m_statPrims, (unsigned long long)m_statBatches, (unsigned long long)m_statDraws,
                 (unsigned long long)m_statSubmits, (unsigned long long)m_statDownloads, (unsigned long long)m_statDownloadPx,
                 (unsigned long long)m_statUploads, (unsigned long long)m_statUploadRows, (unsigned long long)m_statTexUploads,
                 (unsigned long long)m_statTexHits, (unsigned long long)m_statTexRehash, m_targets.size(), m_textures.size());
    std::fprintf(stderr, "[gs:vk] flips=%llu shown from the GPU=%llu%s", (unsigned long long)m_statFlips, (unsigned long long)m_statGpuFlips,
                 m_hostPresent ? "" : " (not shown: no window)");
    if (m_checkPresent)
        std::fprintf(stderr, ", present checks=%llu mismatched=%llu", (unsigned long long)m_statChecks, (unsigned long long)m_statCheckBad);
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "[gs:vk] cpu fallbacks:");
    for (int i = 0; i < FB_COUNT; ++i)
        if (m_statFallback[i])
            std::fprintf(stderr, " %s=%llu", kFallbackNames[i], (unsigned long long)m_statFallback[i]);
    std::fprintf(stderr, "\n[gs:vk] downloads by cause:");
    for (auto &kv : m_statWhy)
        std::fprintf(stderr, " [%s]=%llu", kv.first.c_str(), (unsigned long long)kv.second);
    std::fprintf(stderr, "\n[gs:vk] blend factor > 1 (batches by ALPHA):");
    for (int i = 0; i < 256; ++i)
        if (m_statBigFactor[i])
            std::fprintf(stderr, " %02x=%llu", i, (unsigned long long)m_statBigFactor[i]);
    std::fprintf(stderr, "\n");
}

std::unique_ptr<GSRasterBackend> ps2CreateVulkanGsBackend()
{
    const HostVulkanShared *shared = nullptr;
#if defined(PS2X_HOST_SDL3)
    shared = HostVulkanGetShared();
#endif
    auto backend = std::make_unique<GsVulkanBackend>();
    if (!backend->Create(shared))
        return nullptr;
    return backend;
}
