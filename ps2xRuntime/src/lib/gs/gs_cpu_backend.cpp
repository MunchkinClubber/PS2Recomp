#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cfenv>
#include <chrono>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

using namespace GSInternal;

// Perf counters (read by the SSX3 [ssx3:perf] report): summed raster-worker busy time, and time
// the submitting thread spent blocked on the workers (read-backs, FINISH, full queue).
std::atomic<uint64_t> g_perfGsWorkerNs{0};
std::atomic<uint64_t> g_perfGsWaitNs{0};
std::atomic<uint64_t> g_perfGsSyncs{0};     // read-back / FINISH waits that actually blocked
std::atomic<uint64_t> g_perfGsBarriers{0};  // ordered global commands (transfers, CLUT, clears...)
std::atomic<uint64_t> g_perfGsHazards{0};   // barriers inserted for render-to-texture hazards
std::atomic<uint64_t> g_perfGsDraws{0};
std::atomic<uint64_t> g_perfGsUploadBarriers{0}; // uploads that had to wait behind queued draws

namespace
{
    // Raster band owned by the current thread (see GSCpuBackend::WorkerMain). The synchronous
    // path keeps the defaults: one band covering every row, palette cache 0.
    thread_local uint32_t t_bandIndex = 0u;
    thread_local uint32_t t_bandCount = 1u;
    thread_local uint32_t t_workerSlot = 0u;
    const bool s_gsHazDebug = std::getenv("GS_HAZ_DEBUG") != nullptr; // log barrier causes
    thread_local const uint32_t *t_drawPalette = nullptr; // palette captured when the draw was queued

    inline bool rowInBand(int y)
    {
        return t_bandCount == 1u || ((static_cast<uint32_t>(y) >> 3u) % t_bandCount) == t_bandIndex;
    }
}

#if defined(_MSC_VER)
#define GS_FORCEINLINE __forceinline
#else
#define GS_FORCEINLINE inline __attribute__((always_inline))
#endif

namespace
{
    float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    GS_FORCEINLINE u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    GS_FORCEINLINE u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    GS_FORCEINLINE u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    GS_FORCEINLINE uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    GS_FORCEINLINE int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    GS_FORCEINLINE bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    GS_FORCEINLINE PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    GS_FORCEINLINE TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    bool isFourBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    bool isEightBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
    }

    uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, float fx, float fy)
    {
        const float top = static_cast<float>(c00) + (static_cast<float>(c10) - static_cast<float>(c00)) * fx;
        const float bottom = static_cast<float>(c01) + (static_cast<float>(c11) - static_cast<float>(c01)) * fx;
        return clampU8(static_cast<int>(std::lround(top + (bottom - top) * fy)));
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }
}

GSCpuBackend::GSCpuBackend()
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    Reset();
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    if (vram && vramSize < GSMem::MEMORY_SIZE)
        throw std::invalid_argument("GS CPU backend requires at least 4 MiB of VRAM");

    std::lock_guard<std::mutex> lock(m_mutex);
    SyncUnlocked();
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
}

void GSCpuBackend::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    SyncUnlocked();
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    m_clut.fill(0u);
    m_clutCbp.fill(0u);
    ++m_clutVersion;
    m_texturePageCache.Invalidate();
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || batch.vertexCount == 0u)
        return;
    if (m_threadMode < 0)
        StartWorkersUnlocked();
    if (!threaded())
    {
        // PATH1 draws arrive from inside the VU1 interpreter, which runs with round-toward-zero;
        // rasterise with the default rounding so results do not depend on the submitting path.
        const int rounding = std::fegetround();
        if (rounding != FE_TONEAREST)
            std::fesetround(FE_TONEAREST);
        DrawPrimitive(batch);
        if (rounding != FE_TONEAREST)
            std::fesetround(rounding);
        return;
    }
    g_perfGsDraws.fetch_add(1u, std::memory_order_relaxed);
    NoteDrawHazardsUnlocked(batch);
    Command command;
    command.global = false;
    command.batch = batch;
    const uint8_t tpsm = batch.state.context.tex0.psm;
    if (batch.state.prim.tme && (isFourBitIndexedPsm(tpsm) || isEightBitIndexedPsm(tpsm)))
    {
        // Decode the CLUT on this thread (m_clut is only changed here) and hand the draw an
        // immutable copy, so CLUT loads never have to wait for queued draws.
        const uint64_t key = PaletteKey(batch.state);
        if (!m_sharedPalette || m_sharedPaletteVersion != m_clutVersion || m_sharedPaletteKey != key)
        {
            auto palette = std::make_shared<std::array<uint32_t, 256>>();
            const auto &tex = batch.state.context.tex0;
            const uint32_t n = isFourBitIndexedPsm(tpsm) ? 16u : 256u;
            for (uint32_t i = 0; i < n; ++i)
                (*palette)[i] = LookupCLUT(batch.state, static_cast<uint8_t>(i), tex.cpsm, tex.csm, tex.csa, tex.psm);
            m_sharedPalette = std::move(palette);
            m_sharedPaletteVersion = m_clutVersion;
            m_sharedPaletteKey = key;
        }
        command.palette = m_sharedPalette;
    }
    EnqueueUnlocked(std::move(command));
}

void GSCpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    if (threaded())
    {
        // Runs on this thread: queued draws carry their own decoded palette. Only the VRAM it
        // reads must not be a target of a queued draw (or of a pending global command).
        const uint32_t start = std::min<uint32_t>(tex0.cbp * 256u, 4u * 1024u * 1024u);
        const uint32_t rows = tex0.csm == 0u ? 2u : (static_cast<uint32_t>(texclut.cov) / 32u + 2u);
        const uint32_t width = tex0.csm == 0u ? 1u : std::max<uint32_t>(texclut.cbw, 1u);
        const uint32_t end = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(start) + static_cast<uint64_t>(rows) * width * 8192u, 4u * 1024u * 1024u));
        if (!CanRunDirectUnlocked(start, end))
            SyncUnlocked();
    }
    LoadClutUnlocked(tex0, texclut);
}

void GSCpuBackend::LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    ++m_clutVersion;
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;

    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;

    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u;
        uint32_t sourceY = 0u;
        uint32_t sourceWidth = 1u;

        if (tex0.csm == 0u)
        {
            const uint32_t sourceIndex = swizzleClutIndexCSM1(entry);
            sourceX = sourceIndex & 0x0Fu;
            sourceY = sourceIndex >> 4u;
        }
        else
        {
            sourceWidth = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            sourceX = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
            sourceY = static_cast<uint32_t>(texclut.cov);
        }

        const uint32_t raw = ReadTextureVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY); 
        const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
        if (sixteenBit)
        {
            m_clut[destination] = static_cast<uint16_t>(raw);
        }
        else
        {
            m_clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
            m_clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
        }
    }
}

void GSCpuBackend::Flush()
{
    // Queued work is picked up by the raster workers as soon as it is submitted.
}

void GSCpuBackend::TextureFlush()
{
    // Texture reads go straight to VRAM (no texture cache to invalidate).
}

void GSCpuBackend::Sync(GSSyncReason)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    SyncUnlocked();
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    SyncUnlocked();
    return ReadVramUnlocked(psm, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    if (!m_vram)
        return 0u;

    return GSMem::ReadTexture(nullptr, m_vram, psm, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (threaded())
        EnqueueGlobalUnlocked([this, psm, base, bw, x, y, value]()
                              { WriteVramUnlocked(psm, base, bw, x, y, value); });
    else
        WriteVramUnlocked(psm, base, bw, x, y, value);
}

void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    SyncUnlocked();
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    SyncUnlocked();
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

// ---------------------------------------------------------------------------
// Threaded rasteriser plumbing
// ---------------------------------------------------------------------------

namespace
{
    // Conservative byte range [start, end) of a swizzled buffer: base in blocks, width in 64-pixel
    // units, `rows` pixel rows from the top. Page geometry depends on the pixel format.
    struct GsByteRange
    {
        uint32_t start;
        uint32_t end;
    };
    GsByteRange gsBufferRange(uint32_t baseBlock, uint32_t width64, uint32_t psm, uint32_t rows, uint32_t firstRow = 0u,
                              uint32_t firstCol = 0u, uint32_t lastCol = UINT32_MAX)
    {
        constexpr uint32_t kVram = 4u * 1024u * 1024u;
        uint32_t pageW = 64u, pageH = 32u;
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            pageH = 64u;
            break;
        case GS_PSM_T8:
            pageW = 128u;
            pageH = 64u;
            break;
        case GS_PSM_T4:
            pageW = 128u;
            pageH = 128u;
            break;
        default:
            break;
        }
        const uint32_t pagesW = std::max<uint32_t>((std::max<uint32_t>(width64, 1u) * 64u + pageW - 1u) / pageW, 1u);
        const uint32_t firstPageRow = std::min(firstRow, rows) / pageH;
        const uint32_t pagesH = std::max<uint32_t>((rows + pageH - 1u) / pageH, 1u);
        const uint32_t firstPageCol = std::min(firstCol / pageW, pagesW - 1u);
        const uint32_t lastPageCol = std::min(lastCol / pageW, pagesW - 1u);
        const uint64_t base = static_cast<uint64_t>(baseBlock) * 256u;
        // Pages are row-major: the touched pages lie between (firstRow, firstCol) and (lastRow, lastCol).
        const uint64_t start = base + (static_cast<uint64_t>(firstPageRow) * pagesW + firstPageCol) * 8192u;
        const uint64_t end = base + (static_cast<uint64_t>(pagesH - 1u) * pagesW + lastPageCol + 2u) * 8192u;
        return {static_cast<uint32_t>(std::min<uint64_t>(start, kVram)), static_cast<uint32_t>(std::min<uint64_t>(end, kVram))};
    }

    uint64_t gsNowNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }

    // Short waits only: never sleep (Windows sleep granularity is 1-15 ms, which turned every
    // barrier/sync into a timer tick). Spin, then hand the core over with yield().
    template <class Pred>
    void gsWaitUntil(Pred pred)
    {
        for (int i = 0; i < 1024; ++i)
        {
            if (pred())
                return;
            _mm_pause();
        }
        while (!pred())
        {
            std::this_thread::yield();
            for (int i = 0; i < 16 && !pred(); ++i)
                _mm_pause();
        }
    }
}

GSCpuBackend::~GSCpuBackend()
{
    StopWorkers();
}

void GSCpuBackend::StartWorkersUnlocked()
{
    m_threadMode = 0;
    uint32_t count = 0u;
    if (const char *env = std::getenv("PS2_GS_THREADS"); env && *env)
    {
        count = static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
    }
    else
    {
        const uint32_t hw = std::thread::hardware_concurrency();
        count = hw >= 4u ? std::min<uint32_t>(hw - 2u, 8u) : 0u;
    }
    count = std::min<uint32_t>(count, static_cast<uint32_t>(m_paletteCaches.size() - 1u));
    if (count == 0u)
        return;

    m_ring = std::make_unique<Command[]>(kRingSize);
    m_workerDone = std::make_unique<WorkerSlot[]>(count);
    m_writeIdx.store(0u);
    m_stopWorkers.store(false);
    m_dirty.clear();
    m_threadCount = count;
    m_threadMode = 1;
    for (uint32_t i = 0; i < count; ++i)
        m_workers.emplace_back(&GSCpuBackend::WorkerMain, this, i);
    std::fprintf(stderr, "[gs] threaded rasteriser: %u worker(s) (PS2_GS_THREADS to override, 0 = off)\n", count);
}

void GSCpuBackend::StopWorkers()
{
    if (m_workers.empty())
        return;
    m_stopWorkers.store(true);
    {
        std::lock_guard<std::mutex> lock(m_wakeMutex);
        m_wakeCv.notify_all();
    }
    for (std::thread &worker : m_workers)
        if (worker.joinable())
            worker.join();
    m_workers.clear();
    m_threadCount = 0u;
}

uint64_t GSCpuBackend::MinDone() const
{
    uint64_t result = UINT64_MAX;
    for (uint32_t i = 0; i < m_threadCount; ++i)
        result = std::min(result, m_workerDone[i].done.load(std::memory_order_acquire));
    return result;
}

void GSCpuBackend::WorkerMain(uint32_t index)
{
    t_bandIndex = index;
    t_bandCount = m_threadCount;
    t_workerSlot = index + 1u;
    // Threads can inherit the creator's FP environment (the VU interpreter runs round-toward-zero).
    std::fesetround(FE_TONEAREST);
    uint64_t idx = 0u;
    uint64_t busySince = 0u;
    for (;;)
    {
        if (m_writeIdx.load(std::memory_order_acquire) <= idx)
        {
            if (busySince != 0u)
            {
                g_perfGsWorkerNs.fetch_add(gsNowNs() - busySince, std::memory_order_relaxed);
                busySince = 0u;
            }
            bool ready = false;
            for (int i = 0; i < 2000 && !ready; ++i)
            {
                ready = m_writeIdx.load(std::memory_order_acquire) > idx || m_stopWorkers.load(std::memory_order_relaxed);
                if (!ready)
                    _mm_pause();
            }
            for (int i = 0; i < 200 && !ready; ++i)
            {
                std::this_thread::yield();
                ready = m_writeIdx.load(std::memory_order_acquire) > idx || m_stopWorkers.load(std::memory_order_relaxed);
            }
            if (!ready)
            {
                std::unique_lock<std::mutex> lock(m_wakeMutex);
                m_sleepers.fetch_add(1u);
                m_wakeCv.wait_for(lock, std::chrono::milliseconds(2), [&]()
                                  { return m_writeIdx.load() > idx || m_stopWorkers.load(); });
                m_sleepers.fetch_sub(1u);
            }
            if (m_stopWorkers.load())
                return;
            continue;
        }

        if (busySince == 0u)
            busySince = gsNowNs();
        Command &command = m_ring[idx % kRingSize];
        if (!command.global)
        {
            t_drawPalette = command.palette ? command.palette->data() : nullptr;
            DrawPrimitive(command.batch);
            t_drawPalette = nullptr;
        }
        else if (index == 0u)
        {
            gsWaitUntil([&]()
                        {
                            if (m_stopWorkers.load(std::memory_order_relaxed))
                                return true;
                            for (uint32_t i = 1; i < m_threadCount; ++i)
                                if (m_workerDone[i].done.load(std::memory_order_acquire) < idx)
                                    return false;
                            return true; });
            if (m_stopWorkers.load())
                return;
            if (command.fn)
                command.fn();
            command.fn = nullptr;
        }
        else
        {
            gsWaitUntil([&]()
                        { return m_workerDone[0].done.load(std::memory_order_acquire) > idx ||
                                 m_stopWorkers.load(std::memory_order_relaxed); });
            if (m_stopWorkers.load())
                return;
        }
        m_workerDone[index].done.store(idx + 1u, std::memory_order_release);
        ++idx;
    }
}

void GSCpuBackend::EnqueueUnlocked(Command &&command)
{
    const uint64_t idx = m_writeIdx.load(std::memory_order_relaxed);
    if (idx - MinDone() >= kRingSize)
    {
        const uint64_t t0 = gsNowNs();
        gsWaitUntil([&]()
                    { return idx - MinDone() < kRingSize; });
        g_perfGsWaitNs.fetch_add(gsNowNs() - t0, std::memory_order_relaxed);
    }
    m_ring[idx % kRingSize] = std::move(command);
    m_writeIdx.store(idx + 1u); // seq_cst: pairs with the m_sleepers check below
    if (m_sleepers.load() != 0u)
    {
        std::lock_guard<std::mutex> lock(m_wakeMutex);
        m_wakeCv.notify_all();
    }
}

void GSCpuBackend::EnqueueGlobalUnlocked(std::function<void()> fn, uint32_t touchStart, uint32_t touchEnd)
{
    g_perfGsBarriers.fetch_add(1u, std::memory_order_relaxed);
    Command command;
    command.global = true;
    command.fn = std::move(fn);
    if (!m_epochs.empty())
    {
        const uint64_t done = MinDone();
        m_epochs.erase(std::remove_if(m_epochs.begin(), m_epochs.end(),
                                      [done](const Epoch &e)
                                      { return e.globalIdx < done; }),
                       m_epochs.end());
    }
    Epoch epoch;
    epoch.globalIdx = m_writeIdx.load(std::memory_order_relaxed);
    epoch.ranges = std::move(m_dirty);
    epoch.ranges.insert(epoch.ranges.end(), m_reads.begin(), m_reads.end());
    if (touchEnd > touchStart)
        epoch.ranges.push_back({~0ull, touchStart, touchEnd});
    EnqueueUnlocked(std::move(command));
    m_epochs.push_back(std::move(epoch));
    if (m_epochs.size() > 256u)
    {
        // Bound the bookkeeping: wait for the oldest epochs to retire.
        SyncUnlocked();
        m_epochs.clear();
    }
    m_dirty.clear();
    m_reads.clear();
}

bool GSCpuBackend::CanRunDirectUnlocked(uint32_t start, uint32_t end) const
{
    // True when no queued-but-unfinished command reads or writes [start, end): then this thread
    // may touch that memory right now without changing any result.
    auto hit = [&](const DirtyRange &range)
    { return range.start < end && start < range.end; };
    for (const DirtyRange &range : m_dirty)
        if (hit(range))
            return false;
    for (const DirtyRange &range : m_reads)
        if (hit(range))
            return false;
    if (!m_epochs.empty())
    {
        const uint64_t done = MinDone();
        for (const Epoch &epoch : m_epochs)
        {
            if (epoch.globalIdx < done)
                continue; // finished
            for (const DirtyRange &range : epoch.ranges)
                if (hit(range))
                    return false;
        }
    }
    return true;
}

void GSCpuBackend::SyncUnlocked() const
{
    if (!threaded())
        return;
    const uint64_t target = m_writeIdx.load(std::memory_order_acquire);
    if (MinDone() >= target)
        return;
    g_perfGsSyncs.fetch_add(1u, std::memory_order_relaxed);
    const uint64_t t0 = gsNowNs();
    gsWaitUntil([&]()
                { return MinDone() >= target; });
    g_perfGsWaitNs.fetch_add(gsNowNs() - t0, std::memory_order_relaxed);
}

void GSCpuBackend::NoteDrawHazardsUnlocked(const GSPrimitiveBatch &batch)
{
    // Workers only order draws within their own rows. A draw that reads memory another queued
    // draw writes (render-to-texture), or that aliases a queued target with a different layout,
    // needs every earlier draw finished first: queue a barrier.
    constexpr uint32_t kVramBytes = 4u * 1024u * 1024u;
    const GSDrawState &state = batch.state;
    const GSContext &ctx = state.context;
    const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);

    // Rows the primitive can touch: its vertex y extent clipped to the scissor.
    int yMin = ctx.scissor.y1, yMax = ctx.scissor.y0;
    int xMin = ctx.scissor.x1, xMax = ctx.scissor.x0;
    {
        const int ofx = ctx.xyoffset.ofx >> 4;
        const int ofy = ctx.xyoffset.ofy >> 4;
        const uint32_t count = (state.prim.type == GS_PRIM_SPRITE || state.prim.type == GS_PRIM_LINE || state.prim.type == GS_PRIM_LINESTRIP) ? 2u
                               : (state.prim.type == GS_PRIM_POINT ? 1u : 3u);
        for (uint32_t i = 0; i < count; ++i)
        {
            const float fy = batch.vertices[i].y - static_cast<float>(ofy);
            yMin = std::min(yMin, static_cast<int>(std::floor(fy)) - 1);
            yMax = std::max(yMax, static_cast<int>(std::ceil(fy)) + 1);
            const float fx = batch.vertices[i].x - static_cast<float>(ofx);
            xMin = std::min(xMin, static_cast<int>(std::floor(fx)) - 1);
            xMax = std::max(xMax, static_cast<int>(std::ceil(fx)) + 1);
        }
        xMin = std::max<int>(xMin, ctx.scissor.x0);
        xMax = std::min<int>(xMax, ctx.scissor.x1);
        if (xMax < xMin)
            xMax = xMin;
        yMin = std::max<int>(yMin, ctx.scissor.y0);
        yMax = std::min<int>(yMax, ctx.scissor.y1);
        if (yMax < yMin)
            yMax = yMin;
    }
    auto makeRange = [&](uint64_t key, uint32_t basePage, uint32_t width64, uint32_t psm) -> DirtyRange
    {
        // Rows between yMin and yMax span whole page rows, so the column bounds only tighten the
        // first and last page row; that is what gsBufferRange's row-major bounds assume.
        const GsByteRange r = gsBufferRange(basePage * 32u, width64, psm, static_cast<uint32_t>(yMax) + 1u, static_cast<uint32_t>(yMin),
                                            static_cast<uint32_t>(xMin), static_cast<uint32_t>(xMax));
        return {key, r.start, r.end};
    };
    auto overlaps = [](const DirtyRange &a, const DirtyRange &b)
    { return a.start < b.end && b.start < a.end; };

    DirtyRange targets[2];
    uint32_t targetCount = 0u;
    targets[targetCount++] = makeRange(static_cast<uint64_t>(ctx.frame.fbp) | (static_cast<uint64_t>(fbw) << 16) |
                                           (static_cast<uint64_t>(ctx.frame.psm) << 32),
                                       ctx.frame.fbp, fbw, ctx.frame.psm);
    const uint32_t ztestMethod = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    if (ztestMethod >= 2u || !ctx.zbuf.zmask)
        targets[targetCount++] = makeRange(static_cast<uint64_t>(ctx.zbuf.zbp) | (static_cast<uint64_t>(fbw) << 16) |
                                               (static_cast<uint64_t>(ctx.zbuf.psm) << 32) | (1ull << 40),
                                           ctx.zbuf.zbp, fbw, ctx.zbuf.psm);

    bool barrier = m_dirty.size() >= 16u;
    DirtyRange tex{0u, 0u, 0u};
    if (state.prim.tme)
    {
        const GsByteRange r = gsBufferRange(ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, state.textureHeight);
        tex = {0u, r.start, r.end};
    }
    if (!barrier && state.prim.tme)
    {
        for (const DirtyRange &dirty : m_dirty)
            if (overlaps(dirty, tex))
            {
                barrier = true;
                if (s_gsHazDebug) std::fprintf(stderr, "HAZ tex tbp=%x tbw=%u psm=%x th=%u [%x,%x) vs key=%llx [%x,%x) fbp=%x\n", ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, state.textureHeight, tex.start, tex.end, (unsigned long long)dirty.key, dirty.start, dirty.end, ctx.frame.fbp);
                break;
            }
    }
    for (uint32_t t = 0; t < targetCount && !barrier; ++t)
        for (const DirtyRange &dirty : m_dirty)
            if (dirty.key != targets[t].key && overlaps(dirty, targets[t]))
            {
                if (s_gsHazDebug) std::fprintf(stderr, "HAZ tgt key=%llx [%x,%x) vs key=%llx [%x,%x)\n", (unsigned long long)targets[t].key, targets[t].start, targets[t].end, (unsigned long long)dirty.key, dirty.start, dirty.end);
                barrier = true;
                break;
            }
    if (barrier)
    {
        g_perfGsHazards.fetch_add(1u, std::memory_order_relaxed);
        EnqueueGlobalUnlocked(nullptr, 0u, 0u);
    }

    for (uint32_t t = 0; t < targetCount; ++t)
    {
        bool merged = false;
        for (DirtyRange &dirty : m_dirty)
            if (dirty.key == targets[t].key)
            {
                dirty.start = std::min(dirty.start, targets[t].start);
                dirty.end = std::max(dirty.end, targets[t].end);
                merged = true;
                break;
            }
        if (!merged)
            m_dirty.push_back(targets[t]);
    }

    if (state.prim.tme && tex.end > tex.start)
    {
        bool merged = false;
        for (DirtyRange &read : m_reads)
            if (read.start <= tex.end && tex.start <= read.end)
            {
                read.start = std::min(read.start, tex.start);
                read.end = std::max(read.end, tex.end);
                merged = true;
                break;
            }
        if (!merged)
        {
            if (m_reads.size() >= 32u)
            {
                m_reads.back().start = std::min(m_reads.back().start, tex.start);
                m_reads.back().end = std::max(m_reads.back().end, tex.end);
            }
            else
                m_reads.push_back(tex);
        }
    }
}

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
        DrawSprite(batch);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        DrawTriangle(batch);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        DrawLine(batch);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx = state.context;
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        WritePixel(state, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
        break;
    }
    default:
        break;
    }
}

void GSCpuBackend::WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    const auto &ctx = state.context;
    if (x < ctx.scissor.x0 || x > ctx.scissor.x1 || y < ctx.scissor.y0 || y > ctx.scissor.y1)
        return;
    if (!rowInBand(y))
        return;

    if (state.prim.fge)
    {
        const uint32_t inverseFog = 255u - fog;
        auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
        {
            return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
        };

        r = applyFog(r, state.fogR);
        g = applyFog(g, state.fogG);
        b = applyFog(b, state.fogB);
    }

    const u32 fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const u32 fbw = std::max<u32>(ctx.frame.fbw, 1u);
    const u32 fpsm = ctx.frame.psm;
    const u32 zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const u32 zpsm = ctx.zbuf.psm;

    // The GS clamps Z to the depth buffer format's range (Z24: 0xFFFFFF, Z16: 0xFFFF). Without
    // this, VU output with Z past the range (e.g. negative floats converted to huge values)
    // stored truncated low bits and wrecked later depth tests.
    {
        uint32_t zc = static_cast<uint32_t>(z);
        if (zpsm == GS_PSM_Z24)
            zc = std::min<uint32_t>(zc, 0xFFFFFFu);
        else if (zpsm == GS_PSM_Z16 || zpsm == GS_PSM_Z16S)
            zc = std::min<uint32_t>(zc, 0xFFFFu);
        z = static_cast<int>(zc);
    }

    const PixelWriteMask writeMask = classifyAlphaTest(ctx.test, a, static_cast<uint8_t>(fpsm));
    if (!writeMask.writesAnything())
    {
        return;
    }

    const uint32_t ztestMethod = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    const bool alphaBlendEnabled = state.prim.abe;
    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && fpsm == GS_PSM_CT32;
    const bool destinationAlphaTestNeedsRead = ((ctx.test >> 14) & 0x1u) != 0u && (fpsm == GS_PSM_CT32 || fpsm == GS_PSM_CT16 || fpsm == GS_PSM_CT16S);

    // small optimization, avoid reading the framebuffer for simple draws
    // TODO: only one address lookup for rmw
    const bool frmw = destinationAlphaTestNeedsRead || (writeMask.writesFramebuffer() && ((ctx.frame.fbmsk != 0) || alphaBlendEnabled || preserveDestinationAlpha));

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = ReadVramUnlocked(fpsm, fbp, fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if (bitsPerPixel(fpsm) == 16)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
        else if (fpsm == GS_PSM_CT24)
        {
            // The GS supplies 0x80 as destination alpha for RGB24 blending.
            fbrgba |= 0x80000000u;
        }
    }

    if (!passesDestinationAlphaTest(ctx.test, static_cast<uint8_t>(fpsm), rawFramebufferPixel))
    {
        return;
    }

    bool zpass = false;
    uint32_t storedZ = 0u;
    switch (ztestMethod)
    {
    case 0:
        zpass = false;
        break;
    case 1:
        zpass = true;
        break;
    case 2:
        storedZ = ReadVramUnlocked(zpsm, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) >= storedZ;
        break;
    case 3:
        storedZ = ReadVramUnlocked(zpsm, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) > storedZ;
        break;
    }

    if (!zpass)
    {
        return;
    }

    if (writeMask.writesFramebuffer())
    {
        const u8 srcR = r;
        const u8 srcG = g;
        const u8 srcB = b;

        if (state.prim.abe)
        {
            uint8_t dr = fbrgba & 0xFF;
            uint8_t dg = (fbrgba >> 8) & 0xFF;
            uint8_t db = (fbrgba >> 16) & 0xFF;
            uint8_t da = (fbrgba >> 24) & 0xFF;

            // PABE disables alpha blending when the source alpha MSB is clear.
            if (!(state.pabe && (a & 0x80u) == 0u))
            {
                uint64_t alphaReg = ctx.alpha;
                uint8_t asel = alphaReg & 3;
                uint8_t bsel = (alphaReg >> 2) & 3;
                uint8_t csel = (alphaReg >> 4) & 3;
                uint8_t dsel = (alphaReg >> 6) & 3;
                uint8_t fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);

                auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                int cAlpha = (csel == 0) ? a : (csel == 1) ? da
                                                           : fix;

                r = clampU8(((pickRGB(asel, r, dr) - pickRGB(bsel, r, dr)) * cAlpha >> 7) + pickRGB(dsel, r, dr));
                g = clampU8(((pickRGB(asel, g, dg) - pickRGB(bsel, g, dg)) * cAlpha >> 7) + pickRGB(dsel, g, dg));
                b = clampU8(((pickRGB(asel, b, db) - pickRGB(bsel, b, db)) * cAlpha >> 7) + pickRGB(dsel, b, db));
            }
            else
            {
                r = srcR;
                g = srcG;
                b = srcB;
            }
        }

        if (writeMask.writeAlpha && (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (ctx.frame.fbmsk != 0)
        {
            pixel = (pixel & ~ctx.frame.fbmsk) | (fbrgba & ctx.frame.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (bitsPerPixel(fpsm) == 16)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        WriteVramUnlocked(fpsm, fbp, fbw, x, y, pixel);
    }

    if (writeMask.writeDepth && !ctx.zbuf.zmask)
    {
        WriteVramUnlocked(zpsm, zbp, fbw, x, y, z);
    }
}

uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                  uint8_t index,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    const bool sixteenBit = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;
    const uint32_t sourceIndex = isFourBitIndexedPsm(sourcePsm)
                                     ? (static_cast<uint32_t>(index) & 0x0Fu)
                                     : static_cast<uint32_t>(index);

    uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    if (!sixteenBit && csm == 0u && isEightBitIndexedPsm(sourcePsm))
    {
        const uint32_t block = std::min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }

    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        const uint32_t raw = static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw);
    }
    case GS_PSM_CT24:
    {
        const uint32_t raw = static_cast<uint32_t>(m_clut[clutIndex]) | (static_cast<uint32_t>(m_clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw & 0x00FFFFFFu);
    }
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(m_clut[clutIndex]));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

uint32_t GSCpuBackend::SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v)
{
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;

    const int texW = state.textureWidth;
    const int texH = state.textureHeight;
    const uint64_t clamp = ctx.clamp;
    const uint8_t wrapU = static_cast<uint8_t>(clamp & 0x3u);
    const uint8_t wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    const uint16_t minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    const uint16_t maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    const uint16_t minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    const uint16_t maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);

    float texUf, texVf;
    if (state.prim.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = s * invQ * static_cast<float>(texW);
        texVf = t * invQ * static_cast<float>(texH);
    }

    auto samplePoint = [&](int sampleU, int sampleV) -> uint32_t
    {
        sampleU = wrapTextureCoordinate(sampleU, texW, wrapU, minU, maxU);
        sampleV = wrapTextureCoordinate(sampleV, texH, wrapV, minV, maxV);

        u32 out = ReadTextureVramUnlocked(tex.psm, tex.tbp0, tex.tbw, sampleU, sampleV);

        switch (tex.psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        case GS_PSM_CT24:
        case GS_PSM_Z24:
            return applyTexa(state.texa, tex.psm, out);
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            return applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(out));
        case GS_PSM_T8:
        case GS_PSM_T8H:
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            return LookupCLUT(state, static_cast<u8>(out), tex.cpsm, tex.csm, tex.csa, tex.psm);
        }

        return 0xFFFF00FFu;
    };

    if (!state.linearFilter)
    {
        return samplePoint(static_cast<int>(texUf), static_cast<int>(texVf));
    }

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = static_cast<int>(std::floor(sampleU));
    const int v0 = static_cast<int>(std::floor(sampleV));
    const int u1 = u0 + 1;
    const int v1 = v0 + 1;
    const float fx = sampleU - static_cast<float>(u0);
    const float fy = sampleV - static_cast<float>(v0);

    const uint32_t c00 = samplePoint(u0, v0);
    const uint32_t c10 = samplePoint(u1, v0);
    const uint32_t c01 = samplePoint(u0, v1);
    const uint32_t c11 = samplePoint(u1, v1);

    const uint8_t r = lerpChannel(static_cast<uint8_t>(c00 & 0xFFu),
                                  static_cast<uint8_t>(c10 & 0xFFu),
                                  static_cast<uint8_t>(c01 & 0xFFu),
                                  static_cast<uint8_t>(c11 & 0xFFu),
                                  fx, fy);
    const uint8_t g = lerpChannel(static_cast<uint8_t>((c00 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 8) & 0xFFu),
                                  fx, fy);
    const uint8_t b = lerpChannel(static_cast<uint8_t>((c00 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 16) & 0xFFu),
                                  fx, fy);
    const uint8_t a = lerpChannel(static_cast<uint8_t>((c00 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 24) & 0xFFu),
                                  fx, fy);

    return static_cast<uint32_t>(r) |
           (static_cast<uint32_t>(g) << 8) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(a) << 24);
}

// ---------------------------------------------------------------------------
// Fast pixel pipeline. Everything that only depends on the draw state is
// decoded once per primitive; the per-pixel paths below must stay bit-exact
// with WritePixel / SampleTexture.
// ---------------------------------------------------------------------------
struct GSPixelPipe
{
    int sx0, sx1, sy0, sy1;
    bool fge;
    uint8_t fogR, fogG, fogB;

    uint32_t fbp, fbw, fpsm, zbp, zpsm, zmax;
    uint32_t (*readFb)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
    void (*writeFb)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    uint32_t (*readZ)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
    void (*writeZ)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

    uint64_t test;
    uint32_t ztest;
    bool abe, pabe;
    uint8_t asel, bsel, csel, dsel, fix;
    bool date, dateNeedsRead;
    uint32_t fbmsk;
    bool fbaForce;
    bool fb16, fbCT24, fbCT32;
    bool zmask;
};

struct GSTexSampler
{
    enum Kind : uint8_t
    {
        Direct,
        Texa,
        Texa16,
        Clut,
        Invalid
    };
    uint32_t (*read)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);
    uint32_t tbp0, tbw;
    int texW, texH;
    uint8_t psm;
    Kind kind;
    uint8_t wrapU, wrapV;
    uint16_t minU, maxU, minV, maxV;
    bool fst, linear;
    float texWf, texHf;
    GSTexaReg texa;
    const uint32_t *palette;
};

void GSCpuBackend::SetupPixelPipe(const GSDrawState &state, GSPixelPipe &p) const
{
    const auto &ctx = state.context;
    p.sx0 = ctx.scissor.x0;
    p.sx1 = ctx.scissor.x1;
    p.sy0 = ctx.scissor.y0;
    p.sy1 = ctx.scissor.y1;
    p.fge = state.prim.fge;
    p.fogR = state.fogR;
    p.fogG = state.fogG;
    p.fogB = state.fogB;
    p.fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    p.fbw = std::max<u32>(ctx.frame.fbw, 1u);
    p.fpsm = ctx.frame.psm;
    p.zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    p.zpsm = ctx.zbuf.psm;
    p.zmax = (p.zpsm == GS_PSM_Z24) ? 0xFFFFFFu : ((p.zpsm == GS_PSM_Z16 || p.zpsm == GS_PSM_Z16S) ? 0xFFFFu : 0xFFFFFFFFu);
    p.readFb = m_readVramFuncs[p.fpsm & 0x3Fu];
    p.writeFb = m_writeVramFuncs[p.fpsm & 0x3Fu];
    p.readZ = m_readVramFuncs[p.zpsm & 0x3Fu];
    p.writeZ = m_writeVramFuncs[p.zpsm & 0x3Fu];
    p.test = ctx.test;
    p.ztest = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    p.abe = state.prim.abe;
    p.pabe = state.pabe;
    const uint64_t alphaReg = ctx.alpha;
    p.asel = alphaReg & 3;
    p.bsel = (alphaReg >> 2) & 3;
    p.csel = (alphaReg >> 4) & 3;
    p.dsel = (alphaReg >> 6) & 3;
    p.fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);
    p.date = ((ctx.test >> 14) & 0x1u) != 0u;
    p.dateNeedsRead = p.date && (p.fpsm == GS_PSM_CT32 || p.fpsm == GS_PSM_CT16 || p.fpsm == GS_PSM_CT16S);
    p.fbmsk = ctx.frame.fbmsk;
    p.fbaForce = (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24;
    p.fb16 = bitsPerPixel(static_cast<uint8_t>(p.fpsm)) == 16;
    p.fbCT24 = p.fpsm == GS_PSM_CT24;
    p.fbCT32 = p.fpsm == GS_PSM_CT32;
    p.zmask = ctx.zbuf.zmask;
}

GS_FORCEINLINE void GSCpuBackend::WritePixelFast(const GSPixelPipe &p, int x, int y, uint32_t z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    if (x < p.sx0 || x > p.sx1 || y < p.sy0 || y > p.sy1)
        return;

    if (p.fge)
    {
        const uint32_t inverseFog = 255u - fog;
        r = static_cast<uint8_t>(((static_cast<uint32_t>(fog) * r) >> 8) + ((inverseFog * p.fogR) >> 8));
        g = static_cast<uint8_t>(((static_cast<uint32_t>(fog) * g) >> 8) + ((inverseFog * p.fogG) >> 8));
        b = static_cast<uint8_t>(((static_cast<uint32_t>(fog) * b) >> 8) + ((inverseFog * p.fogB) >> 8));
    }

    if (z > p.zmax)
        z = p.zmax;

    const PixelWriteMask writeMask = classifyAlphaTest(p.test, a, static_cast<uint8_t>(p.fpsm));
    if (!writeMask.writesAnything())
        return;

    const bool writesFb = writeMask.writesFramebuffer();
    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && p.fbCT32;
    const bool frmw = p.dateNeedsRead || (writesFb && ((p.fbmsk != 0) || p.abe || preserveDestinationAlpha));

    uint8_t *const vram = m_vram;
    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = p.readFb(vram, p.fbp, p.fbw, x, y);
        fbrgba = rawFramebufferPixel;
        if (p.fb16)
            fbrgba = Rgba5551ToRgba8888(static_cast<u16>(fbrgba));
        else if (p.fbCT24)
            fbrgba |= 0x80000000u;
    }

    if (p.date && !passesDestinationAlphaTest(p.test, static_cast<uint8_t>(p.fpsm), rawFramebufferPixel))
        return;

    switch (p.ztest)
    {
    case 0:
        return;
    case 1:
        break;
    case 2:
        if (!(z >= p.readZ(vram, p.zbp, p.fbw, x, y)))
            return;
        break;
    case 3:
        if (!(z > p.readZ(vram, p.zbp, p.fbw, x, y)))
            return;
        break;
    }

    if (writesFb)
    {
        if (p.abe && !(p.pabe && (a & 0x80u) == 0u))
        {
            const int dr = fbrgba & 0xFF;
            const int dg = (fbrgba >> 8) & 0xFF;
            const int db = (fbrgba >> 16) & 0xFF;
            const int da = (fbrgba >> 24) & 0xFF;
            const int cAlpha = (p.csel == 0) ? a : (p.csel == 1) ? da : p.fix;
            auto pick = [](uint8_t sel, int cs, int cd) -> int
            { return sel == 0 ? cs : (sel == 1 ? cd : 0); };
            r = clampU8(((pick(p.asel, r, dr) - pick(p.bsel, r, dr)) * cAlpha >> 7) + pick(p.dsel, r, dr));
            g = clampU8(((pick(p.asel, g, dg) - pick(p.bsel, g, dg)) * cAlpha >> 7) + pick(p.dsel, g, dg));
            b = clampU8(((pick(p.asel, b, db) - pick(p.bsel, b, db)) * cAlpha >> 7) + pick(p.dsel, b, db));
        }

        if (writeMask.writeAlpha && p.fbaForce)
            a = static_cast<uint8_t>(a | 0x80u);

        u32 pixel = pack32(r, g, b, a);
        if (p.fbmsk != 0)
            pixel = (pixel & ~p.fbmsk) | (fbrgba & p.fbmsk);
        if (preserveDestinationAlpha)
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        if (p.fb16)
            pixel = Rgba8888ToRgba5551(pixel);
        p.writeFb(vram, p.fbp, p.fbw, x, y, pixel);
    }

    if (writeMask.writeDepth && !p.zmask)
        p.writeZ(vram, p.zbp, p.fbw, x, y, z);
}

uint64_t GSCpuBackend::PaletteKey(const GSDrawState &state) const
{
    const auto &tex = state.context.tex0;
    return static_cast<uint64_t>(tex.cpsm) |
           (static_cast<uint64_t>(tex.csm & 1u) << 8) |
           (static_cast<uint64_t>(tex.csa & 0x1Fu) << 9) |
           (static_cast<uint64_t>(isFourBitIndexedPsm(tex.psm) ? 1u : 0u) << 14) |
           (static_cast<uint64_t>(isEightBitIndexedPsm(tex.psm) ? 1u : 0u) << 15) |
           (static_cast<uint64_t>(state.texa.ta0) << 16) |
           (static_cast<uint64_t>(state.texa.ta1) << 24) |
           (static_cast<uint64_t>(state.texa.aem ? 1u : 0u) << 32);
}

void GSCpuBackend::SetupSampler(const GSDrawState &state, GSTexSampler &s)
{
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;
    s.psm = tex.psm;
    s.read = m_readVramFuncs[tex.psm & 0x3Fu];
    s.tbp0 = tex.tbp0;
    s.tbw = tex.tbw;
    s.texW = state.textureWidth;
    s.texH = state.textureHeight;
    s.texWf = static_cast<float>(s.texW);
    s.texHf = static_cast<float>(s.texH);
    const uint64_t clamp = ctx.clamp;
    s.wrapU = static_cast<uint8_t>(clamp & 0x3u);
    s.wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    s.minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    s.maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    s.minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    s.maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);
    s.fst = state.prim.fst;
    s.linear = state.linearFilter;
    s.texa = state.texa;
    s.palette = nullptr;

    switch (tex.psm)
    {
    case GS_PSM_CT32:
        s.kind = GSTexSampler::Direct;
        break;
    case GS_PSM_Z32:
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        s.kind = GSTexSampler::Texa;
        break;
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        s.kind = GSTexSampler::Texa16;
        break;
    case GS_PSM_T8:
    case GS_PSM_T8H:
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
    {
        s.kind = GSTexSampler::Clut;
        if (t_drawPalette)
        {
            s.palette = t_drawPalette;
            break;
        }
        const bool four = isFourBitIndexedPsm(tex.psm);
        const uint64_t key = PaletteKey(state);
        PaletteCache &cache = m_paletteCaches[t_workerSlot];
        if (cache.version != m_clutVersion || cache.key != key)
        {
            const uint32_t n = four ? 16u : 256u;
            for (uint32_t i = 0; i < n; ++i)
                cache.palette[i] = LookupCLUT(state, static_cast<uint8_t>(i), tex.cpsm, tex.csm, tex.csa, tex.psm);
            cache.version = m_clutVersion;
            cache.key = key;
        }
        s.palette = cache.palette.data();
        break;
    }
    default:
        s.kind = GSTexSampler::Invalid;
        break;
    }
}

GS_FORCEINLINE uint32_t GSCpuBackend::FetchTexel(const GSTexSampler &s, int sampleU, int sampleV) const
{
    sampleU = wrapTextureCoordinate(sampleU, s.texW, s.wrapU, s.minU, s.maxU);
    sampleV = wrapTextureCoordinate(sampleV, s.texH, s.wrapV, s.minV, s.maxV);
    const uint32_t out = s.read(m_vram, s.tbp0, s.tbw, static_cast<uint32_t>(sampleU), static_cast<uint32_t>(sampleV));
    switch (s.kind)
    {
    case GSTexSampler::Direct:
        return out;
    case GSTexSampler::Texa:
        return applyTexa(s.texa, s.psm, out);
    case GSTexSampler::Texa16:
        return applyTexa(s.texa, s.psm, Rgba5551ToRgba8888(static_cast<u16>(out)));
    case GSTexSampler::Clut:
        return s.palette[out & 0xFFu];
    default:
        return 0xFFFF00FFu;
    }
}

namespace
{
    GS_FORCEINLINE uint8_t lerpChannelFast(uint32_t c00, uint32_t c10, uint32_t c01, uint32_t c11, int shift, float fx, float fy)
    {
        const float a = static_cast<float>((c00 >> shift) & 0xFFu);
        const float b = static_cast<float>((c10 >> shift) & 0xFFu);
        const float c = static_cast<float>((c01 >> shift) & 0xFFu);
        const float d = static_cast<float>((c11 >> shift) & 0xFFu);
        const float top = a + (b - a) * fx;
        const float bottom = c + (d - c) * fx;
        const float v = top + (bottom - top) * fy;
        // Equivalent to clampU8(lround(v)) for the value range produced here.
        const double rounded = std::floor(static_cast<double>(v) + 0.5);
        return clampU8(static_cast<int>(rounded));
    }
}

GS_FORCEINLINE uint32_t GSCpuBackend::SampleFast(const GSTexSampler &s, float sv, float tv, float q, uint16_t u, uint16_t v) const
{
    float texUf, texVf;
    if (s.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = sv * invQ * s.texWf;
        texVf = tv * invQ * s.texHf;
    }

    if (!s.linear)
        return FetchTexel(s, static_cast<int>(texUf), static_cast<int>(texVf));

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = static_cast<int>(std::floor(sampleU));
    const int v0 = static_cast<int>(std::floor(sampleV));
    const float fx = sampleU - static_cast<float>(u0);
    const float fy = sampleV - static_cast<float>(v0);

    const uint32_t c00 = FetchTexel(s, u0, v0);
    const uint32_t c10 = FetchTexel(s, u0 + 1, v0);
    const uint32_t c01 = FetchTexel(s, u0, v0 + 1);
    const uint32_t c11 = FetchTexel(s, u0 + 1, v0 + 1);
    if (c00 == c10 && c00 == c01 && c00 == c11)
        return c00;

    return static_cast<uint32_t>(lerpChannelFast(c00, c10, c01, c11, 0, fx, fy)) |
           (static_cast<uint32_t>(lerpChannelFast(c00, c10, c01, c11, 8, fx, fy)) << 8) |
           (static_cast<uint32_t>(lerpChannelFast(c00, c10, c01, c11, 16, fx, fy)) << 16) |
           (static_cast<uint32_t>(lerpChannelFast(c00, c10, c01, c11, 24, fx, fy)) << 24);
}

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;
    u32 z1 = static_cast<u32>(v1.z);

    if (x0 > x1)
        std::swap(x0, x1);
    if (y0 > y1)
        std::swap(y0, y1);

    const int unclippedX0 = x0;
    const int unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);

    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;
    GSPixelPipe pipe;
    SetupPixelPipe(state, pipe);

    if (state.prim.tme)
    {
        const auto &tex = ctx.tex0;
        const int texW = state.textureWidth;
        const int texH = state.textureHeight;

        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u >> 4);
            v0f = static_cast<float>(v0.v >> 4);
            u1f = static_cast<float>(v1.u >> 4);
            v1f = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }

        GSTexSampler sampler;
        SetupSampler(state, sampler);
        float spriteW = static_cast<float>(spanX);
        float spriteH = static_cast<float>(spanY);
        if (spriteW < 1.0f)
            spriteW = 1.0f;
        if (spriteH < 1.0f)
            spriteH = 1.0f;

        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!rowInBand(y))
                continue;
            float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int x = drawX0; x <= drawX1; ++x)
            {
                float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0xFFFF00FFu;
                if (state.prim.fst)
                {
                    const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                    const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                    texel = SampleFast(sampler, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                }
                else
                {
                    texel = SampleFast(sampler, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                WritePixelFast(pipe, x, y, z1, color.r, color.g, color.b, color.a, v1.fog);
            }
        }
    }
    else
    {
        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!rowInBand(y))
                continue;
            for (int x = drawX0; x <= drawX1; ++x)
                WritePixelFast(pipe, x, y, z1, r, g, b, a, v1.fog);
        }
    }
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;

    GSPixelPipe pipe;
    SetupPixelPipe(state, pipe);
    GSTexSampler sampler;
    const bool tme = state.prim.tme;
    const bool fst = state.prim.fst;
    const bool iip = state.prim.iip;
    if (tme)
        SetupSampler(state, sampler);
    const GSTex0Reg &tex = ctx.tex0;

    const float a0 = fy1 - fy2, b0 = fx2 - fx1;
    const float a1 = fy2 - fy0, b1 = fx0 - fx2;

    // Conservative per-row span (in double, widened by a pixel) so that the
    // exact per-pixel edge test below only runs near the triangle.
    const double sd = static_cast<double>(winding) * static_cast<double>(invAbsDenom);
    const double ea0 = a0 * sd, ea1 = a1 * sd;

    for (int y = minY; y <= maxY; ++y)
    {
        if (!rowInBand(y))
            continue;
        const float py = static_cast<float>(y) + 0.5f;
        const float rowB0 = b0 * (py - fy2);
        const float rowB1 = b1 * (py - fy2);

        // w0(px) ~= ea0*(px-fx2) + eb0, w1 similarly, w2 = 1 - w0 - w1.
        double lo = static_cast<double>(minX), hi = static_cast<double>(maxX);
        {
            const double eb0 = static_cast<double>(rowB0) * sd;
            const double eb1 = static_cast<double>(rowB1) * sd;
            const double k[3][2] = {{ea0, eb0 - ea0 * fx2},
                                    {ea1, eb1 - ea1 * fx2},
                                    {-ea0 - ea1, 1.0 - (eb0 - ea0 * fx2) - (eb1 - ea1 * fx2)}};
            bool empty = false;
            for (int e = 0; e < 3 && !empty; ++e)
            {
                const double slope = k[e][0], icpt = k[e][1];
                // Need slope*px + icpt >= -eps (px = x + 0.5); widen by 1e-3 in value space.
                const double limit = -1.0e-3 - icpt;
                if (std::fabs(slope) < 1.0e-12)
                {
                    if (icpt < -1.0e-2)
                        empty = true;
                    continue;
                }
                const double bound = limit / slope - 0.5;
                if (slope > 0.0)
                    lo = std::max(lo, std::floor(bound) - 1.0);
                else
                    hi = std::min(hi, std::ceil(bound) + 1.0);
            }
            if (empty || lo > hi)
                continue;
        }
        const int xs = static_cast<int>(lo);
        const int xe = static_cast<int>(hi);

        for (int x = xs; x <= xe; ++x)
        {
            float px = static_cast<float>(x) + 0.5f;

            float w0 = ((a0 * (px - fx2) + rowB0) * winding) * invAbsDenom;
            float w1 = ((a1 * (px - fx2) + rowB1) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            uint8_t r, g, b, a;
            if (iip)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (tme)
            {
                uint32_t texel;
                if (fst)
                {
                    const uint16_t iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    const uint16_t iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    texel = SampleFast(sampler, 0.0f, 0.0f, 1.0f, iu, iv);
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q
                    // values. Texel coordinates are calculated from S/Q and
                    // T/Q only after interpolation.
                    const float is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    const float it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    const float iq = v0.q * w0 + v1.q * w1 + v2.q * w2;
                    texel = SampleFast(sampler, is, it, iq, 0u, 0u);
                }

                const TextureCombineResult color = combineTexture(tex, r, g, b, a,
                                                                  static_cast<uint8_t>(texel & 0xFF),
                                                                  static_cast<uint8_t>((texel >> 8) & 0xFF),
                                                                  static_cast<uint8_t>((texel >> 16) & 0xFF),
                                                                  static_cast<uint8_t>((texel >> 24) & 0xFF));
                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            const uint8_t fog = clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2));
            WritePixelFast(pipe, x, y, static_cast<u32>(z + 0.5), r, g, b, a, fog);
        }
    }
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        double z = (v0.z + (v1.z - v0.z) * t);
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        WritePixel(state, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
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
        ++step;
    }
}

extern std::atomic<bool> g_ssx3PrimLogArm;         // SSX3 debug (gs_frontend.cpp)
std::atomic<uint32_t> g_ssx3Transfers{0};           // SSX3 debug: transfers started
std::atomic<uint32_t> g_ssx3TransferLogged{0}; // reset by F10 (ps2_runtime.cpp)

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // Threaded mode: m_transfer/m_transferState belong to this thread. Queued upload chunks carry
    // their own copy of the transfer and start position, so transfers never wait on each other.
    if (threaded() && command.direction != 0u)
        SyncUnlocked(); // local->local / local->host read VRAM: let queued work land first
    BeginTransferUnlocked(command);
}

void GSCpuBackend::BeginTransferUnlocked(const GSTransferCommand &command)
{
    g_ssx3Transfers.fetch_add(1u, std::memory_order_relaxed);
    // Log every transfer once armed, plus (from boot) any into the 0x2a00-0x3800 block range the
    // post-intro screen samples its tiles from, except the FMV frame buffer at 0x2a08.
    const uint32_t xdbp = command.bitbltbuf.dbp;
    const bool xInTileRange = xdbp >= 0x2a00u && xdbp < 0x3800u && xdbp != 0x2a08u;
    if ((xInTileRange || g_ssx3PrimLogArm.load(std::memory_order_relaxed)) &&
        g_ssx3TransferLogged.fetch_add(1u, std::memory_order_relaxed) < 200u)
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "[ssx3:xfer] dir=%u dbp=0x%x dbw=%u dpsm=0x%x dsa=(%u,%u) sbp=0x%x sbw=%u spsm=0x%x ssa=(%u,%u) size=%ux%u",
                      command.direction, command.bitbltbuf.dbp, command.bitbltbuf.dbw, command.bitbltbuf.dpsm,
                      command.trxpos.dsax, command.trxpos.dsay, command.bitbltbuf.sbp, command.bitbltbuf.sbw,
                      command.bitbltbuf.spsm, command.trxpos.ssax, command.trxpos.ssay, command.trxreg.rrw, command.trxreg.rrh);
        RUNTIME_LOG(buf << std::endl);
    }
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!data || sizeBytes == 0u || !m_vram)
        return;
    if (threaded())
    {
        // Write straight into VRAM from this thread when no queued draw reads or writes the
        // destination; otherwise queue it in order behind those draws.
        const GSTransferCommand &t = m_transfer;
        if (m_transferState.direction != 0u || t.trxreg.rrw == 0u || t.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
            return;
        const GsByteRange r = gsBufferRange(t.bitbltbuf.dbp, t.bitbltbuf.dbw, t.bitbltbuf.dpsm,
                                            static_cast<uint32_t>(t.trxpos.dsay) + static_cast<uint32_t>(t.trxreg.rrh));
        if (CanRunDirectUnlocked(r.start, r.end))
        {
            UploadImageUnlocked(data, sizeBytes);
            return;
        }
        // Queue the chunk behind the draws that still use that memory, with its own copy of the
        // transfer and start position; this thread's transfer state advances right away.
        g_perfGsUploadBarriers.fetch_add(1u, std::memory_order_relaxed);
        auto copy = std::make_shared<std::vector<uint8_t>>(data, data + sizeBytes);
        const GSTransferCommand xfer = m_transfer;
        const GSTransferSnapshot start = m_transferState;
        UploadImageImpl(m_transfer, m_transferState, data, sizeBytes, false);
        EnqueueGlobalUnlocked([this, copy, xfer, start]()
                              {
                                  GSTransferSnapshot st = start;
                                  UploadImageImpl(xfer, st, copy->data(), static_cast<uint32_t>(copy->size()), true); },
                              r.start, r.end);
    }
    else
        UploadImageUnlocked(data, sizeBytes);
}

void GSCpuBackend::UploadImageUnlocked(const uint8_t *data, uint32_t sizeBytes)
{
    UploadImageImpl(m_transfer, m_transferState, data, sizeBytes, true);
}

// Writes one chunk of a host->local transfer described by `xfer`, advancing `st`. With
// write=false only `st` advances (the producer's bookkeeping for a chunk handed to the queue).
void GSCpuBackend::UploadImageImpl(const GSTransferCommand &xfer, GSTransferSnapshot &st,
                                   const uint8_t *data, uint32_t sizeBytes, bool write)
{
    auto writePx = [&](uint32_t psm, uint32_t bp, uint32_t bw, uint32_t x, uint32_t y, uint32_t v)
    {
        if (write)
            WriteVramUnlocked(psm, bp, bw, x, y, v);
    };
    if (!data || sizeBytes == 0u || !m_vram || st.direction != 0u)
        return;
    if (xfer.trxreg.rrw == 0u || xfer.trxreg.rrh == 0u || st.totalPixels == 0u)
        return;

    const uint32_t dbp = xfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(xfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = xfer.bitbltbuf.dpsm;
    const uint32_t rrw = xfer.trxreg.rrw;
    const uint32_t dsax = xfer.trxpos.dsax;
    uint32_t offset = 0u;

    auto advancePixel = [&](uint32_t count)
    {
        const uint32_t totalPixels = st.totalPixels;
        st.copiedPixels =
            std::min<uint32_t>(totalPixels, st.copiedPixels + count);

        if (st.copiedPixels >= totalPixels)
        {
            st.direction = 3u;
            st.totalPixels = 0u;
            return;
        }

        st.x = dsax + (st.copiedPixels % rrw);
        st.y = xfer.trxpos.dsay + (st.copiedPixels / rrw);
    };

    while (offset < sizeBytes && st.direction == 0u)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            writePx(dpsm, dbp, dbw, st.x, st.y, value);
            offset += 4u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            writePx(dpsm, dbp, dbw, st.x, st.y, value);
            offset += 3u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            writePx(dpsm, dbp, dbw, st.x, st.y, value);
            offset += 2u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            writePx(dpsm, dbp, dbw, st.x, st.y, data[offset++]);
            advancePixel(1u);
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            const uint32_t firstPixel = st.copiedPixels;
            writePx(dpsm, dbp, dbw,
                              dsax + (firstPixel % rrw),
                              xfer.trxpos.dsay + (firstPixel / rrw),
                              packed & 0x0Fu);
            if (firstPixel + 1u < st.totalPixels)
            {
                const uint32_t secondPixel = firstPixel + 1u;
                writePx(dpsm, dbp, dbw,
                                  dsax + (secondPixel % rrw),
                                  xfer.trxpos.dsay + (secondPixel / rrw),
                                  (packed >> 4u) & 0x0Fu);
            }
            advancePixel(std::min<uint32_t>(2u, st.totalPixels - firstPixel));
            break;
        }
        default:
            return;
        }
    }
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    SyncUnlocked();
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || context.frame.fbw == 0u)
        return false;
    const uint8_t cpsm = context.frame.psm;
    if (cpsm != GS_PSM_CT32 && cpsm != GS_PSM_CT24 && cpsm != GS_PSM_CT16 && cpsm != GS_PSM_CT16S)
        return false;
    if (threaded())
        EnqueueGlobalUnlocked([this, context, rgba]()
                              { ClearFramebufferUnlocked(context, rgba); });
    else
        ClearFramebufferUnlocked(context, rgba);
    return true;
}

void GSCpuBackend::ClearFramebufferUnlocked(const GSContext &context, uint32_t rgba)
{

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return;
    }
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    SnapshotVram(snapshot);
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(displayFrame1, origin1, width1, height1, false, true, selected1, crt1, preferred1) &&
            copySource(displayFrame2, origin2, width2, height2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < height1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            if (fieldMode)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    if (!copySource(displayFrame, origin, result.width, result.height, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    if (fieldMode)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}
