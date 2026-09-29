#include "runtime/vu1_native.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/ps2_memory.h"

#include <algorithm>
#include <atomic>
#include <set>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

extern std::atomic<uint32_t> g_ssx3Xgkicks; // ps2_vu1_core.cpp
std::atomic<uint64_t> g_vu1NativeRuns{0};
std::atomic<uint64_t> g_vu1NativeHandoffs{0};
std::atomic<uint64_t> g_vu1NativeMisses{0};
std::atomic<uint64_t> g_vu0NativeRuns{0};
std::atomic<uint64_t> g_vu0NativeMisses{0};

// Friend of VU1Interpreter: the translated programs' view of the interpreter internals.
struct VU1NativeAccess
{
    static uint8_t *xgkickBuffer(VU1Interpreter &vu) { return vu.m_xgkick.packet.data(); }
    static uint32_t xgkickBufferSize() { return VU1Interpreter::XgkickPipeline::kBufferSize; }
    static uint64_t &cycle(VU1Interpreter &vu) { return vu.m_cycle; }
    static uint32_t workingClip(VU1Interpreter &vu) { return vu.m_workingClip; }

    static void submit(VU1NativeCtx &c)
    {
        g_ssx3Xgkicks.fetch_add(1u, std::memory_order_relaxed);
        if (c.memory)
            c.memory->submitGifPacket(GifPathId::Path1, c.xgPacket, c.xgTotal);
        else if (c.gs)
            c.gs->processGIFPacket(c.xgPacket, c.xgTotal);
    }

    static void reservedStop(VU1Interpreter &vu, uint32_t code)
    {
        vu.reportReservedInstruction(false, code);
    }

    // Load the native pipeline state into the interpreter and let it run on from `pc`.
    static void handoff(VU1NativeCtx &c, uint32_t pc, bool branchPending, uint32_t target, bool ebit)
    {
        VU1Interpreter &vu = *c.vu;
        vu1n::xgProgressTo(c, c.cyc);
        vu.m_cycle = c.cyc;
        vu.m_state.cycles = c.cyc;
        vu.m_state.pc = pc;
        vu.m_state.branchPending = branchPending;
        vu.m_state.branchTarget = target;
        vu.m_state.branchDelay = 0u;
        vu.m_state.ebit = ebit;
        vu.m_state.haltAfterDelaySlot = false;
        for (uint32_t r = 0; r < 32; ++r)
            for (uint32_t l = 0; l < 4; ++l)
                vu.m_vfReady[r][l] = c.vfReady[r][l];
        for (uint32_t r = 0; r < 16; ++r)
            vu.m_viReady[r] = c.viReady[r];
        for (uint32_t l = 0; l < 4; ++l)
            vu.m_accReady[l] = c.accReady[l];
        // Register values already hold their latest writes; readers stall until those land.
        vu.m_fdiv = {};
        if (c.qPending)
        {
            vu.m_fdiv.valid = true;
            vu.m_fdiv.readyCycle = c.qReady;
            vu.m_fdiv.value = c.qValue;
            vu.m_fdiv.statusDi = c.qDi;
        }
        vu.m_efuMask = 0u;
        for (uint32_t i = 0; i < 2; ++i)
        {
            vu.m_efu[i] = {};
            if (c.pPending[i])
            {
                vu.m_efu[i].valid = true;
                vu.m_efu[i].readyCycle = c.pReady[i];
                vu.m_efu[i].value = c.pValue[i];
                vu.m_efuMask |= 1u << i;
            }
        }
        vu.m_efuResourceReady = c.efuResourceReady;
        vu.m_flagMask = 0u;
        for (auto &entry : vu.m_flagPipeline)
            entry = {};
        for (uint32_t i = 0, slot = 0; i < c.flagCount && slot < vu.m_flagPipeline.size(); ++i)
        {
            const VU1NativeCtx::FlagEntry &e = c.flags[(c.flagHead + i) % VU1NativeCtx::kFlagSlots];
            if (e.kind == 0u)
                continue;
            auto &entry = vu.m_flagPipeline[slot];
            entry.valid = true;
            entry.issueCycle = e.issue;
            entry.readyCycle = e.ready;
            entry.mac = e.mac;
            entry.status = e.status;
            entry.extraSticky = e.sticky;
            entry.clip = e.clip;
            entry.writesMac = (e.kind & 1u) != 0u;
            entry.writesStatus = (e.kind & 2u) != 0u;
            entry.writesSticky = (e.kind & 4u) != 0u;
            entry.writesClip = (e.kind & 8u) != 0u;
            vu.m_flagMask |= 1u << slot;
            ++slot;
        }
        vu.m_workingClip = c.workingClip;
        vu.m_viBranchBackupValid = c.bkValid;
        vu.m_viBranchBackupReg = c.bkReg;
        vu.m_viBranchBackupValue = c.bkVal;
        vu.resetXgkickState();
        if (c.xgActive)
        {
            vu.m_xgkick.active = true;
            vu.m_xgkick.sourceAddress = c.xgSource;
            vu.m_xgkick.copiedBytes = c.xgCopied;
            vu.m_xgkick.currentTagEnd = c.xgTagEnd;
            vu.m_xgkick.currentTagEop = c.xgTagEop;
            vu.m_xgkick.totalBytes = c.xgTotal;
            vu.m_xgkick.issueCycle = c.xgIssue;
            // The interpreter copies when its credit reaches 2 at a boundary; c.xgNext is the
            // next copying boundary, reached after (xgNext - cyc) more increments.
            vu.m_xgkick.cycleCredit = (c.xgNext > c.cyc + 1u) ? 0u : 1u;
        }
        vu.m_nextCommitCycle = 0u; // re-scan on the next commit
    }
};

namespace vu1n
{
    namespace
    {
        std::mutex &registryMutex()
        {
            static std::mutex m;
            return m;
        }
        std::vector<const VU1NativeImage *> &registry()
        {
            static std::vector<const VU1NativeImage *> r;
            return r;
        }

        uint64_t hashSpans(const uint8_t *code, const uint16_t *spans)
        {
            uint64_t h = 1469598103934665603ull;
            for (const uint16_t *s = spans; s[0] != 0u || s[1] != 0u; s += 2)
                for (uint32_t i = s[0]; i < s[1]; ++i)
                {
                    h ^= code[i];
                    h *= 1099511628211ull;
                }
            return h;
        }
    }

    void registerImage(const VU1NativeImage *image)
    {
        std::lock_guard<std::mutex> lock(registryMutex());
        registry().push_back(image);
    }

    void xgStart(VU1NativeCtx &c, uint32_t qwordAddress)
    {
        c.xgActive = true;
        c.xgSource = (qwordAddress * 16u) % c.memSize;
        c.xgCopied = 0u;
        c.xgTagEnd = 0u;
        c.xgTagEop = false;
        c.xgTotal = 0u;
        c.xgIssue = c.cyc;
        c.xgNext = c.cyc + 1u; // XGKICK's issue cycle already counts toward PATH1
    }

    // Same per-qword work as VU1Interpreter::progressXgkick.
    void xgProgressTo(VU1NativeCtx &c, uint64_t boundary)
    {
        while (c.xgActive && c.xgNext <= boundary)
        {
            c.xgNext += 2u;
            if (c.xgCopied > c.xgBufSize - 16u)
            {
                c.xgActive = false;
                c.stopped = true;
                return;
            }
            const uint32_t qwordOffset = c.xgCopied;
            const uint32_t first = (c.xgSource + c.xgCopied) % c.memSize;
            if (first + 16u <= c.memSize)
                std::memcpy(c.xgPacket + c.xgCopied, c.mem + first, 16u);
            else
                for (uint32_t i = 0; i < 16u; ++i)
                    c.xgPacket[c.xgCopied + i] = c.mem[(c.xgSource + c.xgCopied + i) % c.memSize];
            c.xgCopied += 16u;

            if (c.xgTagEnd == 0u)
            {
                uint64_t tagLo = 0;
                std::memcpy(&tagLo, c.xgPacket + qwordOffset, sizeof(tagLo));
                const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
                const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
                uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
                if (nreg == 0u)
                    nreg = 16u;
                uint64_t tagBytes = 16u;
                if (format == 0u)
                    tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
                else if (format == 1u)
                    tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
                else if (format == 2u)
                    tagBytes += static_cast<uint64_t>(nloop) * 16u;
                else
                {
                    c.xgActive = false;
                    c.stopped = true;
                    return;
                }
                if (tagBytes > c.xgBufSize - qwordOffset)
                {
                    c.xgActive = false;
                    c.stopped = true;
                    return;
                }
                c.xgTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
                c.xgTagEop = ((tagLo >> 15) & 1u) != 0u;
                if (c.xgTagEop)
                    c.xgTotal = c.xgTagEnd;
            }

            if (c.xgCopied >= c.xgTagEnd)
            {
                if (c.xgTagEop)
                {
                    VU1NativeAccess::submit(c);
                    c.xgActive = false;
                }
                else
                {
                    c.xgTagEnd = 0u;
                    c.xgTagEop = false;
                }
            }
        }
    }

    void finish(VU1NativeCtx &c)
    {
        // VU1Interpreter::flushPipelines: advance until nothing is pending.
        uint64_t end = c.cyc;
        for (uint32_t r = 1; r < 32; ++r)
            for (uint32_t l = 0; l < 4; ++l)
                end = std::max(end, c.vfReady[r][l]);
        for (uint32_t r = 1; r < 16; ++r)
            end = std::max(end, c.viReady[r]);
        for (uint32_t l = 0; l < 4; ++l)
            end = std::max(end, c.accReady[l]);
        if (c.qPending)
            end = std::max(end, c.qReady);
        end = std::max(end, pAllReady(c));
        for (uint32_t i = 0; i < c.flagCount; ++i)
            end = std::max(end, c.flags[(c.flagHead + i) % VU1NativeCtx::kFlagSlots].ready);
        if (c.xgActive)
        {
            xgProgressTo(c, c.cyc);
            while (c.xgActive)
            {
                ++c.cyc;
                xgProgressTo(c, c.cyc);
            }
            end = std::max(end, c.cyc);
        }
        c.cyc = end;
        commitAll(c);
        commitP(c);
    }

    VU1NativeExit handoff(VU1NativeCtx &c, uint32_t pc, bool branchPending, uint32_t target, bool ebit)
    {
        g_vu1NativeHandoffs.fetch_add(1u, std::memory_order_relaxed);
        static const bool logHandoffs = std::getenv("PS2_VU1_NATIVE_LOG") != nullptr;
        if (logHandoffs)
            std::fprintf(stderr, "[vu1n] handoff at pc=0x%X branch=%d target=0x%X ebit=%d cyc=%llu\n", pc, branchPending ? 1 : 0, target, ebit ? 1 : 0,
                         static_cast<unsigned long long>(c.cyc));
        VU1NativeAccess::handoff(c, pc, branchPending, target, ebit);
        return VU1NativeExit::Handoff;
    }

}

extern int g_vu1NativeGeneratedImages; // generated translation units (keep them linked)
extern int g_vu0NativeGeneratedImages;

namespace
{
    // Programs that ran without a translation are appended to vu1_programs.bin (input format of
    // tools/vu1recomp) so they can be translated later. PS2_VU1_DUMP=0 turns this off.
    void dumpUntranslated(const uint8_t *code, uint32_t startPC, bool vu0)
    {
        static const bool enabled = []
        {
            const char *v = std::getenv("PS2_VU1_DUMP");
            return !(v && *v == '0');
        }();
        if (!enabled)
            return;
        static std::mutex mutex;
        static std::set<std::pair<uint64_t, uint32_t>> seen;
        static uint32_t written[2] = {0u, 0u};
        std::lock_guard<std::mutex> lock(mutex);
        const uint32_t size = vu0 ? PS2_VU0_CODE_SIZE : PS2_VU1_CODE_SIZE;
        const char *file = vu0 ? "vu0_programs.bin" : "vu1_programs.bin";
        uint64_t h = vu0 ? 0x9e3779b97f4a7c15ull : 1469598103934665603ull;
        for (uint32_t i = 0; i < size; ++i)
        {
            h ^= code[i];
            h *= 1099511628211ull;
        }
        if (!seen.insert({h, startPC}).second || written[vu0 ? 1 : 0] >= 512u)
            return;
        if (FILE *f = std::fopen(file, "ab"))
        {
            const uint32_t hdr[2] = {startPC, 1u};
            std::fwrite(hdr, sizeof(hdr), 1, f);
            std::fwrite(code, 1, size, f);
            std::fclose(f);
            ++written[vu0 ? 1 : 0];
            std::fprintf(stderr, "[vu1n] no translation for %s program at 0x%X (code %016llx) - saved to %s\n",
                         vu0 ? "VU0" : "VU1", startPC, static_cast<unsigned long long>(h), file);
        }
    }
}

bool g_vu1NativeEnabled = []
{
    const char *v = std::getenv("PS2_VU1_NATIVE");
    return !(v && *v == '0');
}();

// Run `startPC` natively if a translated image matches the current micro memory. On handoff the
// interpreter's run() continues with the state loaded by VU1NativeAccess::handoff.
bool vu1NativeLookupAndRun(VU1Interpreter &vu, uint8_t *vuCode, uint8_t *vuData, uint32_t dataSize,
                           GS &gs, PS2Memory *memory, uint32_t startPC, uint32_t maxCycles, bool &handedOff)
{
    using namespace vu1n;
    handedOff = false;
    if (!g_vu1NativeEnabled || !memory)
        return false;
    const bool vu0 = vuCode == memory->getVU0Code();
    if (!vu0 && vuCode != memory->getVU1Code())
        return false;
    // Untranslated programs are still dumped when no images exist yet (to seed the generator).
    const bool haveImages = (vu0 ? g_vu0NativeGeneratedImages : g_vu1NativeGeneratedImages) != 0;

    // Per entry PC: the matched image plus a copy of the code it was translated from. The game
    // re-uploads microcode many times a frame (bumping the generation), usually unchanged, so a
    // memcmp against the copy revalidates cheaply; only real changes pay for a hash lookup.
    struct Entry
    {
        const VU1NativeImage *image = nullptr;
        std::vector<uint8_t> bytes;
        uint64_t validated = ~0ull;
    };
    thread_local std::unordered_map<uint32_t, Entry> caches[2]; // VU1, VU0
    thread_local const uint8_t *cacheCodes[2] = {nullptr, nullptr};
    auto &cache = caches[vu0 ? 1 : 0];
    if (cacheCodes[vu0 ? 1 : 0] != vuCode)
    {
        cache.clear();
        cacheCodes[vu0 ? 1 : 0] = vuCode;
    }
    const uint64_t generation = vu0 ? memory->getVU0CodeGeneration() : memory->getVU1CodeGeneration();
    Entry &entry = cache[startPC];
    if (entry.validated != generation)
    {
        bool same = false;
        if (entry.image)
        {
            same = true;
            size_t offset = 0;
            for (const uint16_t *sp = entry.image->spans; (sp[0] != 0u || sp[1] != 0u) && same; sp += 2)
            {
                const size_t len = static_cast<size_t>(sp[1] - sp[0]);
                same = std::memcmp(vuCode + sp[0], entry.bytes.data() + offset, len) == 0;
                offset += len;
            }
        }
        if (!same)
        {
            entry.image = nullptr;
            entry.bytes.clear();
            std::lock_guard<std::mutex> lock(registryMutex());
            for (const VU1NativeImage *candidate : registry())
            {
                if (candidate->vu0 != vu0)
                    continue;
                bool hasEntry = false;
                for (const uint16_t *e = candidate->entries; *e != 0xFFFFu; ++e)
                    if (*e == startPC)
                        hasEntry = true;
                if (hasEntry && hashSpans(vuCode, candidate->spans) == candidate->hash)
                {
                    entry.image = candidate;
                    for (const uint16_t *sp = candidate->spans; sp[0] != 0u || sp[1] != 0u; sp += 2)
                        entry.bytes.insert(entry.bytes.end(), vuCode + sp[0], vuCode + sp[1]);
                    break;
                }
            }
            if (!entry.image)
                dumpUntranslated(vuCode, startPC, vu0); // once per code change and entry point
        }
        entry.validated = generation;
    }
    const VU1NativeImage *image = entry.image;
    if (!image || !haveImages)
    {
        (vu0 ? g_vu0NativeMisses : g_vu1NativeMisses).fetch_add(1u, std::memory_order_relaxed);
        return false;
    }

    VU1NativeCtx c;
    c.vu = &vu;
    c.st = &vu.state();
    c.mem = vuData;
    c.memSize = dataSize;
    c.cyc = VU1NativeAccess::cycle(vu);
    c.budgetEnd = c.cyc + maxCycles;
    for (auto &r : c.vfReady)
        for (auto &l : r)
            l = 0;
    c.workingClip = VU1NativeAccess::workingClip(vu);
    c.xgPacket = VU1NativeAccess::xgkickBuffer(vu);
    c.xgBufSize = VU1NativeAccess::xgkickBufferSize();
    c.gs = &gs;
    c.memory = memory;

    (vu0 ? g_vu0NativeRuns : g_vu1NativeRuns).fetch_add(1u, std::memory_order_relaxed);
    // Same floating-point environment as VU1Interpreter::run.
    const int previousRounding = std::fegetround();
    const bool vuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    const VU1NativeExit exit = image->fn(c, startPC);
    if (vuRounding && previousRounding != -1)
        std::fesetround(previousRounding);
    if (exit == VU1NativeExit::Handoff)
    {
        handedOff = true;
        return true;
    }
    VU1NativeAccess::cycle(vu) = c.cyc;
    c.st->cycles = c.cyc;
    if (c.stopped)
        VU1NativeAccess::reservedStop(vu, 0xFFFFFFF0u);
    return true;
}
