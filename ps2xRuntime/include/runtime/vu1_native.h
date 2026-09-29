#ifndef PS2_VU1_NATIVE_H
#define PS2_VU1_NATIVE_H

// Statically recompiled VU1 microprograms.
//
// ps2xRuntime/tools/vu1recomp translates captured VU1 code images into C++ (one function per
// image, entered with the MSCAL start PC). The generated code keeps the interpreter's timing
// model: every register lane carries the cycle its latest write lands, reads stall until then,
// and the cycle count drives everything that can observe time (clip flags, Q/P, XGKICK's
// transfer progress). Writes are applied immediately because a read always stalls until the
// latest write to what it reads has landed; MAC/status flags are not produced (no translated
// program reads them). Anything the translation cannot follow (a JR to an unknown target, a
// branch in a delay slot, the cycle budget) hands the exact pipeline state to the interpreter,
// which continues from there.

#include "runtime/ps2_vu1.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

class GS;
class PS2Memory;

struct VU1NativeCtx
{
    VU1State *st = nullptr;
    uint8_t *mem = nullptr; // VU1 data memory
    uint32_t memSize = 0;
    uint64_t cyc = 0;
    uint64_t budgetEnd = 0;

    uint64_t vfReady[32][4]{};
    uint64_t viReady[16]{};
    uint64_t accReady[4]{};

    // FDIV (Q)
    bool qPending = false;
    uint64_t qReady = 0;
    float qValue = 0.0f;
    uint32_t qDi = 0;

    // EFU (P): two result slots, committed in slot order
    bool pPending[2]{};
    uint64_t pReady[2]{};
    float pValue[2]{};
    uint64_t efuResourceReady = 0;

    // Clip flag pipeline (FIFO, becomes visible 4 cycles after issue)
    uint32_t workingClip = 0;
    uint32_t clipHead = 0, clipCount = 0;
    uint64_t clipReady[8]{};
    uint64_t clipIssue[8]{};
    uint32_t clipValue[8]{};

    // Branch VI read-back of the previous pair's integer write
    bool bkValid = false;
    uint8_t bkReg = 0;
    int32_t bkVal = 0;

    // XGKICK (PATH1): one qword copied every other cycle boundary
    bool xgActive = false;
    uint32_t xgSource = 0, xgCopied = 0, xgTagEnd = 0, xgTotal = 0;
    bool xgTagEop = false;
    uint64_t xgNext = 0; // next cycle boundary that copies a qword
    uint64_t xgIssue = 0;
    uint8_t *xgPacket = nullptr;
    uint32_t xgBufSize = 0;

    VU1Interpreter *vu = nullptr;
    GS *gs = nullptr;
    PS2Memory *memory = nullptr;
    bool stopped = false; // an error the interpreter would also have stopped on
};

// Result of a native run.
enum class VU1NativeExit : uint8_t
{
    Ended,   // E-bit program end (pipelines flushed)
    Handoff, // continue in the interpreter from st->pc with the pipeline state handed over
};

using VU1NativeProgramFn = VU1NativeExit (*)(VU1NativeCtx &ctx, uint32_t startPc);

struct VU1NativeImage
{
    const char *name;
    uint64_t hash;             // FNV-1a over the translated pairs listed in `spans`
    const uint16_t *spans;     // [start, end) byte pairs, terminated by {0, 0}
    const uint16_t *entries;   // entry PCs, terminated by 0xFFFF
    VU1NativeProgramFn fn;
};

#if defined(_MSC_VER)
#define VU1N_INLINE __forceinline
#else
#define VU1N_INLINE inline __attribute__((always_inline))
#endif

namespace vu1n
{
    // ---- registry -------------------------------------------------------------------------
    void registerImage(const VU1NativeImage *image);
    struct Registrar
    {
        explicit Registrar(const VU1NativeImage *image) { registerImage(image); }
    };

    // ---- helpers used by generated code ----------------------------------------------------
    VU1N_INLINE float N(float v)
    {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        const uint32_t e = (b >> 23) & 0xFFu;
        if (e == 0u)
            b &= 0x80000000u;
        else if (e == 0xFFu)
            b = (b & 0x80000000u) | 0x7F7FFFFFu;
        std::memcpy(&v, &b, 4);
        return v;
    }

    VU1N_INLINE float normResult(float v)
    {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        const uint32_t e = (b >> 23) & 0xFFu;
        if ((b & 0x7FFFFFFFu) == 0u)
            return v;
        if (e == 0u)
            b &= 0x80000000u;
        else if (e == 0xFFu)
            b = (b & 0x80000000u) | 0x7F7FFFFFu;
        std::memcpy(&v, &b, 4);
        return v;
    }

    VU1N_INLINE uint32_t expOf(float v)
    {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        return (b >> 23) & 0xFFu;
    }

    // Value side of VU1Interpreter::normalizeFmacExactResult.
    inline float fixExact(float r, long double e)
    {
        const bool negative = std::signbit(e);
        const long double mag = std::fabs(e);
        uint32_t bits = negative ? 0x80000000u : 0u;
        float out = r;
        if (mag == 0.0L || mag < static_cast<long double>(std::numeric_limits<float>::min()))
            std::memcpy(&out, &bits, 4);
        else if (mag > static_cast<long double>(std::numeric_limits<float>::max()))
        {
            bits |= 0x7F7FFFFFu;
            std::memcpy(&out, &bits, 4);
        }
        return out;
    }

    // r = a + b (or a - b, a * b) in float; the exact result only matters when r is zero,
    // denormal or near the float limits.
    VU1N_INLINE float fAdd(float a, float b)
    {
        const float r = a + b;
        const uint32_t e = expOf(r);
        if (e >= 2u && e <= 253u)
            return r;
        return fixExact(r, static_cast<long double>(a) + static_cast<long double>(b));
    }
    VU1N_INLINE float fSub(float a, float b)
    {
        const float r = a - b;
        const uint32_t e = expOf(r);
        if (e >= 2u && e <= 253u)
            return r;
        return fixExact(r, static_cast<long double>(a) - static_cast<long double>(b));
    }
    VU1N_INLINE float fMul(float a, float b)
    {
        const float r = a * b;
        const uint32_t e = expOf(r);
        if (e >= 2u && e <= 253u)
            return r;
        return fixExact(r, static_cast<long double>(a) * static_cast<long double>(b));
    }
    // acc + a*b / acc - a*b: skip the exact path when no cancellation or range problem is possible.
    VU1N_INLINE float fMadd(float acc, float a, float b)
    {
        const float p = a * b;
        const float r = acc + p;
        const uint32_t er = expOf(r), ep = expOf(p);
        if (er >= 3u && er <= 252u && ep <= 253u && er + 17u >= ep)
            return r;
        return fixExact(r, static_cast<long double>(acc) + static_cast<long double>(a) * static_cast<long double>(b));
    }
    VU1N_INLINE float fMsub(float acc, float a, float b)
    {
        const float p = a * b;
        const float r = acc - p;
        const uint32_t er = expOf(r), ep = expOf(p);
        if (er >= 3u && er <= 252u && ep <= 253u && er + 17u >= ep)
            return r;
        return fixExact(r, static_cast<long double>(acc) - static_cast<long double>(a) * static_cast<long double>(b));
    }

    inline int32_t floatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

    VU1N_INLINE uint32_t fbits(float v)
    {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        return b;
    }
    VU1N_INLINE float bitsf(uint32_t b)
    {
        float v;
        std::memcpy(&v, &b, 4);
        return v;
    }

    VU1N_INLINE void stall(VU1NativeCtx &c, uint64_t ready)
    {
        if (ready > c.cyc)
            c.cyc = ready;
    }

    // Commit pipelines whose results are visible at the current cycle.
    VU1N_INLINE void commitQ(VU1NativeCtx &c)
    {
        if (c.qPending && c.qReady <= c.cyc)
        {
            c.st->q = c.qValue;
            c.qPending = false;
        }
    }
    inline void commitP(VU1NativeCtx &c)
    {
        for (int i = 0; i < 2; ++i)
            if (c.pPending[i] && c.pReady[i] <= c.cyc)
            {
                c.st->p = c.pValue[i];
                c.pPending[i] = false;
            }
    }
    VU1N_INLINE void commitClip(VU1NativeCtx &c)
    {
        while (c.clipCount != 0u && c.clipReady[c.clipHead] <= c.cyc)
        {
            c.st->clip = c.clipValue[c.clipHead];
            c.clipHead = (c.clipHead + 1u) & 7u;
            --c.clipCount;
        }
    }
    inline void pushClip(VU1NativeCtx &c, uint32_t value)
    {
        commitClip(c);
        const uint32_t slot = (c.clipHead + c.clipCount) & 7u;
        c.clipReady[slot] = c.cyc + 4u;
        c.clipIssue[slot] = c.cyc;
        c.clipValue[slot] = value;
        if (c.clipCount < 8u)
            ++c.clipCount;
        else
            c.stopped = true;
    }
    inline void queueClip(VU1NativeCtx &c, uint32_t bits6)
    {
        c.workingClip = ((c.workingClip << 6) | (bits6 & 0x3Fu)) & 0xFFFFFFu;
        pushClip(c, c.workingClip);
    }

    inline void queueQ(VU1NativeCtx &c, float value, uint32_t latency, uint32_t di)
    {
        commitQ(c);
        c.qPending = true;
        c.qReady = c.cyc + latency;
        c.qValue = normResult(value);
        c.qDi = di & 0x30u;
    }
    inline void queueP(VU1NativeCtx &c, float value, uint32_t latency)
    {
        commitP(c);
        value = normResult(value);
        for (int i = 0; i < 2; ++i)
            if (!c.pPending[i])
            {
                c.pPending[i] = true;
                c.pReady[i] = c.cyc + latency;
                c.pValue[i] = value;
                c.efuResourceReady = c.cyc + (latency > 0u ? latency - 1u : 0u);
                return;
            }
        c.stopped = true;
    }
    inline uint64_t pAllReady(const VU1NativeCtx &c)
    {
        uint64_t r = 0;
        for (int i = 0; i < 2; ++i)
            if (c.pPending[i] && c.pReady[i] > r)
                r = c.pReady[i];
        return r;
    }

    // Branch read of a VI register (the pair right after an integer write sees the old value).
    VU1N_INLINE int32_t brVi(const VU1NativeCtx &c, uint8_t reg)
    {
        if (reg == 0u)
            return 0;
        if (c.bkValid && c.bkReg == reg)
            return c.bkVal;
        return c.st->vi[reg];
    }

    // XGKICK support (vu1_native.cpp)
    void xgProgressTo(VU1NativeCtx &c, uint64_t boundary);
    void xgStart(VU1NativeCtx &c, uint32_t qwordAddress);
    VU1N_INLINE void xgBeforeStore(VU1NativeCtx &c)
    {
        if (c.xgActive)
            xgProgressTo(c, c.cyc);
    }
    // Stall a new XGKICK until the previous transfer has finished.
    inline void xgWaitIdle(VU1NativeCtx &c)
    {
        if (!c.xgActive)
            return;
        xgProgressTo(c, c.cyc);
        while (c.xgActive)
        {
            ++c.cyc;
            xgProgressTo(c, c.cyc);
        }
    }

    VU1N_INLINE void storeQ(VU1NativeCtx &c, uint32_t addr, const uint32_t words[4], uint8_t dest)
    {
        xgBeforeStore(c);
        uint8_t *p = c.mem + addr;
        if (dest & 8u)
            std::memcpy(p + 0, &words[0], 4);
        if (dest & 4u)
            std::memcpy(p + 4, &words[1], 4);
        if (dest & 2u)
            std::memcpy(p + 8, &words[2], 4);
        if (dest & 1u)
            std::memcpy(p + 12, &words[3], 4);
    }

    // Program end: let every pipeline drain like VU1Interpreter::flushPipelines.
    void finish(VU1NativeCtx &c);
    // Hand the exact pipeline state to the interpreter; it resumes at `pc`. With branchPending,
    // `pc` is a branch delay slot and the interpreter jumps to `target` after it.
    VU1NativeExit handoff(VU1NativeCtx &c, uint32_t pc, bool branchPending = false, uint32_t target = 0u, bool ebit = false);

    // EFU approximations (same as ps2_vu1_lower.cpp)
    inline float eatan(float value)
    {
        constexpr float k[] = {0.999999344348907f, -0.333298563957214f, 0.199465364217758f, -0.13085337519646f,
                               0.096420042216778f, -0.055909886956215f, 0.021861229091883f, -0.004054057877511f};
        const float squared = value * value;
        float polynomial = k[7];
        for (int index = 6; index >= 0; --index)
            polynomial = k[index] + squared * polynomial;
        return 0.785398185253143f + value * polynomial;
    }
    inline float esin(float value)
    {
        constexpr float k[] = {1.0f, -0.166666567325592f, 0.008333025500178f, -0.000198074136279f, 0.000002601886990f};
        const float squared = value * value;
        float polynomial = k[4];
        for (int index = 3; index >= 0; --index)
            polynomial = k[index] + squared * polynomial;
        return value * polynomial;
    }
    inline float eexp(float value)
    {
        constexpr float k[] = {0.249998688697815f, 0.031257584691048f, 0.002591371303424f,
                               0.000171562001924f, 0.000005430199963f, 0.000000690600018f};
        float polynomial = k[5];
        for (int index = 4; index >= 0; --index)
            polynomial = k[index] + value * polynomial;
        polynomial = 1.0f + value * polynomial;
        polynomial *= polynomial;
        polynomial *= polynomial;
        return polynomial != 0.0f ? 1.0f / polynomial : std::numeric_limits<float>::max();
    }
}

// Called by VU1Interpreter::execute for VU1: runs a translated program if one matches the code
// currently in VU1 micro memory. Returns false if there is none (the interpreter runs it). With
// handedOff set, the interpreter must continue (run()) from the state that was loaded into it.
bool vu1NativeLookupAndRun(VU1Interpreter &vu, uint8_t *vuCode, uint8_t *vuData, uint32_t dataSize,
                           GS &gs, PS2Memory *memory, uint32_t startPC, uint32_t maxCycles, bool &handedOff);

#endif
