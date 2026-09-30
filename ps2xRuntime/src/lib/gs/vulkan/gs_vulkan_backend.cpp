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

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
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
    using PageSet = std::bitset<512>;
    constexpr uint32_t kTargetHeight = 1024u;
    constexpr uint32_t kRingSize = 64u << 20;
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
            for (uint32_t c = c0; c <= c1 && c < ppr + 1u; ++c)
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
            break;
        default:
            d.fallback = FB_PRIM;
            return d;
        }
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

        if (st.prim.tme)
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
    bool Create();

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
        uint64_t lastUse = 0; // m_submitSerial of the last command buffer using it
        uint8_t maxAlpha = 0; // largest texel alpha
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
    static PageSet displayPages(const GSPresentationRequest &request);

    // ---- drawing ----
    void appendPrimitive(const GSPrimitiveBatch &batch);
    void flushBatch();
    Texture *getTexture(const GSDrawState &st, uint32_t &decW, uint32_t &decH, uint32_t &texFlags);
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
    GpuImage m_dstImg;                   // copy of the target for draws that read it (set 1)
    VkDescriptorSet m_dstSet = VK_NULL_HANDLE;
    bool ensureDstImage(uint32_t width);
    VkPipelineLayout m_pipeLayout = VK_NULL_HANDLE;
    VkShaderModule m_vs = VK_NULL_HANDLE, m_fs = VK_NULL_HANDLE;
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    std::unordered_map<uint32_t, VkPipeline> m_pipelines;
    GpuImage m_dummyTex;
    VkDescriptorSet m_dummySet = VK_NULL_HANDLE;
    std::vector<GpuImage> m_deferredImages;
    std::vector<VkDescriptorSet> m_deferredSets;

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
    uint64_t m_statFlips = 0;
    uint64_t m_statDstCopies = 0, m_statAliasCopies = 0;
    bool m_noAlias = false;
    const char *m_why = "?";
    std::unordered_map<std::string, uint64_t> m_statWhy;
    uint64_t m_statBigFactor[256]{};
};

// ------------------------------------------------------------------------------------------
// Vulkan setup
// ------------------------------------------------------------------------------------------

GsVulkanBackend::~GsVulkanBackend()
{
    if (!m_device)
    {
        if (m_instance)
            m_it.vkDestroyInstance ? m_it.vkDestroyInstance(m_instance, nullptr) : void();
        return;
    }
    if (m_stats)
        printStats();
    m_dt.vkDeviceWaitIdle(m_device);
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
    for (GpuImage &img : m_deferredImages)
        destroyImage(img);
    destroyImage(m_dummyTex);
    destroyImage(m_dstImg);
    for (auto &kv : m_pipelines)
        m_dt.vkDestroyPipeline(m_device, kv.second, nullptr);
    if (m_descPool) m_dt.vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    if (m_vs) m_dt.vkDestroyShaderModule(m_device, m_vs, nullptr);
    if (m_fs) m_dt.vkDestroyShaderModule(m_device, m_fs, nullptr);
    if (m_pipeLayout) m_dt.vkDestroyPipelineLayout(m_device, m_pipeLayout, nullptr);
    if (m_setLayout) m_dt.vkDestroyDescriptorSetLayout(m_device, m_setLayout, nullptr);
    if (m_sampler) m_dt.vkDestroySampler(m_device, m_sampler, nullptr);
    if (m_ring) m_dt.vkDestroyBuffer(m_device, m_ring, nullptr);
    if (m_ringMem) m_dt.vkFreeMemory(m_device, m_ringMem, nullptr);
    if (m_readback) m_dt.vkDestroyBuffer(m_device, m_readback, nullptr);
    if (m_readbackMem) m_dt.vkFreeMemory(m_device, m_readbackMem, nullptr);
    if (m_fence) m_dt.vkDestroyFence(m_device, m_fence, nullptr);
    if (m_cmdPool) m_dt.vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
    m_dt.vkDestroyDevice(m_device, nullptr);
    m_it.vkDestroyInstance(m_instance, nullptr);
}

bool GsVulkanBackend::Create()
{
    m_stats = envFlag("PS2_GS_VK_STATS");
    m_noAlias = envFlag("PS2_GS_VK_NOALIAS");
    m_cpu.SetSynchronous();
    m_presenter.SetSynchronous();
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
    if (m_dt.vkAllocateCommandBuffers(m_device, &cai, &m_cmd) != VK_SUCCESS)
        return false;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (m_dt.vkCreateFence(m_device, &fci, nullptr, &m_fence) != VK_SUCCESS)
        return false;
    void *mapped = nullptr;
    if (!createBuffer(kRingSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
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

void GsVulkanBackend::submitAndWait()
{
    if (!m_cmdOpen)
        return;
    endRendering();
    m_dt.vkEndCommandBuffer(m_cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_cmd;
    m_dt.vkQueueSubmit(m_queue, 1, &si, m_fence);
    m_dt.vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    m_dt.vkResetFences(m_device, 1, &m_fence);
    m_dt.vkResetCommandBuffer(m_cmd, 0);
    m_cmdOpen = false;
    m_ringOffset = 0;
    ++m_submitSerial;
    ++m_statSubmits;
    for (GpuImage &img : m_deferredImages)
        destroyImage(img);
    m_deferredImages.clear();
    if (!m_deferredSets.empty())
    {
        m_dt.vkFreeDescriptorSets(m_device, m_descPool, static_cast<uint32_t>(m_deferredSets.size()), m_deferredSets.data());
        m_deferredSets.clear();
    }
}

uint32_t GsVulkanBackend::ringAlloc(uint32_t size, uint32_t align)
{
    uint32_t off = (m_ringOffset + align - 1u) & ~(align - 1u);
    if (off + size > kRingSize)
    {
        submitAndWait();
        off = 0;
    }
    m_ringOffset = off + size;
    beginCmd();
    return off;
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
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
    else
        createImage(t->img, w, kTargetHeight, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT);
    const uint32_t groups = kTargetHeight / pageDims(psm).h;
    t->stale = groups >= 64u ? ~0ull : ((1ull << groups) - 1ull);
    m_targets.push_back(t);
    if (m_stats)
        std::fprintf(stderr, "[gs:vk] target fbp=0x%x fbw=%u psm=0x%x (%ux%u)\n", fbp, fbw, psm, w, kTargetHeight);
    return *t;
}

uint64_t GsVulkanBackend::groupsFor(const Target &t, const PageSet &pages) const
{
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
    for (uint32_t p = 0; p < 512u; ++p)
        if (pages.test(p))
            m_pageSerial[p] = m_serial;
}

void GsVulkanBackend::ensureVramCurrent(const PageSet &pages)
{
    for (size_t i = 0; i < m_targets.size(); ++i)
    {
        Target &t = *m_targets[i];
        if (t.dirty && (t.dirtyPages & pages).any())
            download(t);
    }
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

void GsVulkanBackend::downloadAll(bool includeDepth)
{
    for (size_t i = 0; i < m_targets.size(); ++i)
        if (includeDepth || !m_targets[i]->depth)
            download(*m_targets[i]);
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
    ensureVramCurrent(groupPages(t, stale));
    t.stale &= ~stale;
    const uint32_t groups = kTargetHeight / d.h;
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
        if (!g && !(o.dirty && (o.dirtyPages & drawPages).any()))
            continue;
        if (o.dirty && (o.dirtyPages & drawPages).any())
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
    const uint32_t cw = std::min(w, src->img.width), ch = std::min(h, src->img.height);
    m_why = "texture";
    for (size_t i = 0; i < m_targets.size(); ++i)
    {
        Target &o = *m_targets[i];
        if (&o != src && o.dirty && (o.dirtyPages & pages).any())
            download(o);
    }
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
        const uint32_t n = m_cpu.DecodePalette(st, palette);
        key.palette = hashWords(palette, n) ^ n;
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
            if (t->dirty && (t->dirtyPages & pages).any() && logged < 40)
            {
                ++logged;
                std::fprintf(stderr, "[gs:vk] rt-as-texture: tex tbp=%x tbw=%u psm=%x %ux%u clamp=%llx linear=%d <- target fbp=%x fbw=%u psm=%x dirty=(%d,%d)-(%d,%d)\n",
                             tex.tbp0, tex.tbw, tex.psm, w, h, (unsigned long long)ctx.clamp, st.linearFilter, t->fbp, t->fbw, t->psm, t->dx0, t->dy0, t->dx1, t->dy1);
            }
    }
    ensureVramCurrent(pages);

    auto it = m_textures.find(key);
    if (it != m_textures.end())
    {
        Texture &t = it->second;
        bool valid = true;
        for (uint16_t p : t.pages)
            if (m_pageSerial[p] > t.serial)
            {
                valid = false;
                break;
            }
        if (valid)
        {
            t.lastUse = m_submitSerial;
            ++m_statTexHits;
            return &t;
        }
    }

    m_decodeBuf.resize(static_cast<size_t>(w) * h);
    if (!m_cpu.DecodeTexture(st, w, h, isIndexedPsm(tex.psm) ? palette : nullptr, m_decodeBuf.data()))
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
                m_deferredImages.push_back(e->second.img);
                m_deferredSets.push_back(e->second.set);
                m_textures.erase(e);
            }
        }
        it = m_textures.emplace(key, Texture{}).first;
    }
    Texture &t = it->second;
    if (t.img.width != w || t.img.height != h || !t.img.image)
    {
        if (t.img.image)
            m_deferredImages.push_back(t.img);
        t.img = GpuImage{};
        createImage(t.img, w, h, VK_FORMAT_R8G8B8A8_UINT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
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
    // Re-uploading an image the recorded commands still sample is fine: the copy is ordered
    // after them by the barrier.
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
    tt.pages.clear();
    for (uint32_t p = 0; p < 512u; ++p)
        if (pages.test(p))
            tt.pages.push_back(static_cast<uint16_t>(p));
    tt.serial = m_serial;
    {
        uint32_t ma = 0;
        for (uint32_t v : m_decodeBuf)
            ma = std::max(ma, v >> 24);
        tt.maxAlpha = static_cast<uint8_t>(ma);
    }
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
    const GSDrawState &st = b.state;
    const GSContext &ctx = st.context;
    const DrawSetup &setup = b.setup;
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
    if (st.prim.tme)
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
        ++m_statBigFactor[ctx.alpha & 0xFFu];

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
    if (st.prim.tme)
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
    if (st.prim.iip && st.prim.type != GS_PRIM_SPRITE)
        flags |= F_IIP;
    if (ctx.frame.psm == GS_PSM_CT16)
        flags |= F_CT16;
    if (depth && ctx.zbuf.psm != GS_PSM_Z32)
        flags |= F_ZROUND;
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
    flushBatch();
    PageSet pages;
    // CSM1: a 16x16 (or 8x2) block-ordered CLUT occupies at most four blocks from CBP.
    pages.set((tex0.cbp >> 5) & 511u);
    pages.set(((tex0.cbp + 3u) >> 5) & 511u);
    if (tex0.csm)
        addRectPages(pages, tex0.cbp, texclut.cbw, tex0.cpsm, texclut.cou * 16, texclut.cov, texclut.cou * 16 + 255, texclut.cov + 1);
    m_why = "clut";
    ensureVramCurrent(pages);
    m_cpu.LoadClut(tex0, texclut);
}

void GsVulkanBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
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
    m_why = "flip";
    ensureVramCurrent(displayPages(request));
    ++m_statFlips;
    if (m_stats && (m_statFlips % 300u) == 0u)
        printStats();
    std::lock_guard<std::mutex> plock(m_presentMutex);
    m_snapLatest.assign(m_vram, m_vram + m_vramSize);
    m_snapRequest = request;
    m_haveSnap = true;
    m_lastFlipNs.store(nowNs(), std::memory_order_relaxed);
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
    std::fprintf(stderr,
                 "[gs:vk] alias copies=%llu dst copies=%llu prims=%llu batches=%llu draws=%llu submits=%llu downloads=%llu (%llu px) uploads=%llu (%llu rows) tex uploads=%llu hits=%llu targets=%zu textures=%zu\n",
                 (unsigned long long)m_statAliasCopies, (unsigned long long)m_statDstCopies, (unsigned long long)m_statPrims, (unsigned long long)m_statBatches, (unsigned long long)m_statDraws,
                 (unsigned long long)m_statSubmits, (unsigned long long)m_statDownloads, (unsigned long long)m_statDownloadPx,
                 (unsigned long long)m_statUploads, (unsigned long long)m_statUploadRows, (unsigned long long)m_statTexUploads,
                 (unsigned long long)m_statTexHits, m_targets.size(), m_textures.size());
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
    auto backend = std::make_unique<GsVulkanBackend>();
    if (!backend->Create())
        return nullptr;
    return backend;
}
