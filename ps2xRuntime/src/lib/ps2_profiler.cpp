// Built-in sampling profiler (Windows). Set PS2_PROFILE=1: a background thread samples the
// instruction pointer of every thread in the process about once per millisecond and every
// 10 seconds writes profile_NNN.txt with the hottest functions per thread for that window
// (symbols from ps2EntryRunner.pdb via DbgHelp).
//
// Also here: the stutter log (runtime/ps2_hitch.h) - "[slow]" lines for waits that took long and
// "[hitch]" reports for frames that came late. With the profiler on, a hitch report lists what
// every thread was executing while the late frame was being made: the sampler keeps each thread's
// last few seconds of samples for that.

#include "runtime/ps2_hitch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

void ps2ProfilerStartIfEnabled();

namespace
{
    struct HitchRequest
    {
        uint64_t beginNs = 0, endNs = 0;
    };
    std::mutex s_hitchMutex; // the lines of the log below, and s_hitchRequests
    std::vector<HitchRequest> s_hitchRequests;
    std::atomic<bool> s_samplerOn{false};

    uint64_t hitchOriginNs()
    {
        static const uint64_t origin = ps2HitchNowNs();
        return origin;
    }
    const uint64_t s_hitchOriginInit = hitchOriginNs(); // (at program start)

    double hitchSeconds(uint64_t ns)
    {
        const uint64_t origin = hitchOriginNs();
        return ns > origin ? static_cast<double>(ns - origin) / 1e9 : 0.0;
    }
}

uint64_t ps2HitchNowNs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

void ps2SlowLog(const char *thread, const char *what, uint64_t ns)
{
    static const bool s_off = std::getenv("PS2_SLOWLOG") && std::getenv("PS2_SLOWLOG")[0] == '0';
    if (s_off)
        return;
    const uint64_t now = ps2HitchNowNs();
    std::lock_guard<std::mutex> lock(s_hitchMutex);
    static uint64_t windowStart = 0;
    static uint32_t shown = 0, dropped = 0;
    if (now - windowStart > 2000000000ull)
    {
        if (dropped)
            std::fprintf(stderr, "[slow] (%u more lines not shown)\n", dropped);
        windowStart = now;
        shown = dropped = 0;
    }
    if (shown >= 40u)
    {
        ++dropped;
        return;
    }
    ++shown;
    std::fprintf(stderr, "[slow] t=%.3f %s: %s: %.1f ms\n", hitchSeconds(now), thread, what, static_cast<double>(ns) / 1e6);
}

void ps2HitchReport(const char *what, uint64_t beginNs, uint64_t endNs)
{
    static const bool s_off = std::getenv("PS2_SLOWLOG") && std::getenv("PS2_SLOWLOG")[0] == '0';
    if (s_off)
        return;
    std::lock_guard<std::mutex> lock(s_hitchMutex);
    // (the game thread's reports and the renderer's are limited separately)
    static uint64_t lastNs[2] = {0, 0};
    static uint32_t dropped[2] = {0, 0};
    const int kind = what[0] == 'g' ? 0 : 1;
    if (lastNs[kind] != 0u && endNs < lastNs[kind] + 150000000ull)
    {
        ++dropped[kind];
        return;
    }
    lastNs[kind] = endNs;
    if (dropped[kind])
        std::fprintf(stderr, "[hitch] t=%.3f %s (and %u more like it since the last report)\n", hitchSeconds(endNs), what, dropped[kind]);
    else
        std::fprintf(stderr, "[hitch] t=%.3f %s\n", hitchSeconds(endNs), what);
    dropped[kind] = 0;
    if (s_samplerOn.load(std::memory_order_relaxed) && s_hitchRequests.size() < 16u)
        s_hitchRequests.push_back(HitchRequest{beginNs, endNs});
}

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <dbghelp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma comment(lib, "dbghelp.lib")

namespace
{
    std::string symbolFor(HANDLE process, DWORD64 address, std::unordered_map<DWORD64, std::string> &cache)
    {
        auto it = cache.find(address);
        if (it != cache.end())
            return it->second;
        alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512];
        SYMBOL_INFO *symbol = reinterpret_cast<SYMBOL_INFO *>(buffer);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = 511;
        DWORD64 displacement = 0;
        std::string name;
        if (SymFromAddr(process, address, &displacement, symbol))
        {
            name = symbol->Name;
            // Past the end of that symbol (or its size is not known and it is far away): the
            // address belongs to code without a name of its own - a label in an assembly file
            // is then the nearest one before it ("_NLG_Return2" collected a sixth of the VU1
            // thread's samples that way). Say where it really is: source file and line.
            if (displacement >= (symbol->Size != 0u ? symbol->Size : 0x100u))
            {
                IMAGEHLP_LINE64 line{};
                line.SizeOfStruct = sizeof(line);
                DWORD lineDisplacement = 0;
                char text[512];
                if (SymGetLineFromAddr64(process, address, &lineDisplacement, &line) && line.FileName)
                {
                    const char *file = std::strrchr(line.FileName, '\\');
                    std::snprintf(text, sizeof(text), "(no name; %s:%lu, after %s)", file ? file + 1 : line.FileName, static_cast<unsigned long>(line.LineNumber), symbol->Name);
                }
                else
                    std::snprintf(text, sizeof(text), "(no name; %s+0x%llx)", symbol->Name, static_cast<unsigned long long>(displacement & ~0xFFull));
                name = text;
            }
        }
        else
        {
            char module[MAX_PATH] = "?";
            HMODULE handle = nullptr;
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(address), &handle) &&
                handle)
            {
                GetModuleFileNameA(handle, module, MAX_PATH);
                const char *slash = std::strrchr(module, '\\');
                char text[MAX_PATH + 32];
                std::snprintf(text, sizeof(text), "%s+0x%llx", slash ? slash + 1 : module,
                              static_cast<unsigned long long>(address - reinterpret_cast<DWORD64>(handle)));
                name = text;
            }
            else
                name = "?";
        }
        cache.emplace(address, name);
        return name;
    }

    // The thread's description (SetThreadDescription: "GS thread", ...), or "thread <id>".
    std::string threadName(HANDLE thread, DWORD tid)
    {
        using GetThreadDescriptionFn = HRESULT(WINAPI *)(HANDLE, PWSTR *);
        static const GetThreadDescriptionFn getDescription =
            reinterpret_cast<GetThreadDescriptionFn>(GetProcAddress(GetModuleHandleW(L"Kernel32.dll"), "GetThreadDescription"));
        char text[128];
        text[0] = 0;
        if (getDescription && thread)
        {
            PWSTR wide = nullptr;
            if (getDescription(thread, &wide) >= 0 && wide)
            {
                if (WideCharToMultiByte(CP_UTF8, 0, wide, -1, text, static_cast<int>(sizeof(text)), nullptr, nullptr) <= 0)
                    text[0] = 0;
                LocalFree(wide);
            }
        }
        if (!text[0])
            std::snprintf(text, sizeof(text), "thread %lu", static_cast<unsigned long>(tid));
        return text;
    }

    void samplerMain()
    {
        const DWORD self = GetCurrentThreadId();
        const DWORD pid = GetCurrentProcessId();
        HANDLE process = GetCurrentProcess();
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        SymInitialize(process, nullptr, TRUE);

        struct ThreadInfo
        {
            HANDLE handle = nullptr;
            std::unordered_map<DWORD64, uint32_t> hits;
            uint64_t samples = 0;
            // the last kRecent samples (time, instruction pointer), for hitch reports
            std::vector<std::pair<uint64_t, DWORD64>> recent;
            size_t recentAt = 0;
        };
        constexpr size_t kRecent = 4096;
        std::unordered_map<DWORD, ThreadInfo> threads;
        std::unordered_map<DWORD64, std::string> names;
        auto lastEnum = std::chrono::steady_clock::now() - std::chrono::seconds(10);
        auto windowStart = std::chrono::steady_clock::now();
        int fileIndex = 0;

        for (;;)
        {
            const auto now = std::chrono::steady_clock::now();
            if (now - lastEnum > std::chrono::seconds(1))
            {
                lastEnum = now;
                HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                if (snap != INVALID_HANDLE_VALUE)
                {
                    THREADENTRY32 te{};
                    te.dwSize = sizeof(te);
                    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te))
                    {
                        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self || threads.count(te.th32ThreadID))
                            continue;
                        HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
                        if (h)
                            threads[te.th32ThreadID].handle = h;
                    }
                    CloseHandle(snap);
                }
            }

            const uint64_t sampleNs = ps2HitchNowNs();
            for (auto &[tid, info] : threads)
            {
                if (!info.handle)
                    continue;
                if (SuspendThread(info.handle) == static_cast<DWORD>(-1))
                {
                    CloseHandle(info.handle);
                    info.handle = nullptr;
                    continue;
                }
                alignas(16) CONTEXT context{};
                context.ContextFlags = CONTEXT_CONTROL;
                const BOOL got = GetThreadContext(info.handle, &context);
                ResumeThread(info.handle);
                // Record only after resuming: the target may hold a heap lock.
                if (got)
                {
                    ++info.hits[context.Rip];
                    ++info.samples;
                    if (info.recent.size() < kRecent)
                        info.recent.emplace_back(sampleNs, context.Rip);
                    else
                    {
                        info.recent[info.recentAt] = {sampleNs, context.Rip};
                        info.recentAt = (info.recentAt + 1u) % kRecent;
                    }
                }
            }

            // Hitch reports asked for since the last round: per thread, where its samples from
            // that time were.
            std::vector<HitchRequest> requests;
            {
                std::lock_guard<std::mutex> lock(s_hitchMutex);
                requests.swap(s_hitchRequests);
            }
            for (const HitchRequest &request : requests)
            {
                std::string text;
                for (auto &[tid, info] : threads)
                {
                    std::unordered_map<std::string, uint32_t> byName;
                    uint32_t total = 0;
                    for (const auto &sample : info.recent)
                        if (sample.first >= request.beginNs && sample.first <= request.endNs)
                        {
                            ++byName[symbolFor(process, sample.second, names)];
                            ++total;
                        }
                    if (total == 0u)
                        continue;
                    std::vector<std::pair<std::string, uint32_t>> top(byName.begin(), byName.end());
                    std::sort(top.begin(), top.end(), [](const auto &a, const auto &b)
                              { return a.second != b.second ? a.second > b.second : a.first < b.first; });
                    const std::string name = threadName(info.handle, tid);
                    // (threads without a name that only waited: the pool threads of the system)
                    if (top.size() == 1u && name.compare(0, 7, "thread ") == 0 &&
                        (top[0].first.find("Wait") != std::string::npos || top[0].first.find("GetMessage") != std::string::npos))
                        continue;
                    text += "[hitch]   " + name + ":";
                    for (size_t i = 0; i < top.size() && i < 6u; ++i)
                    {
                        char item[320];
                        std::snprintf(item, sizeof(item), "%s %.0f%% %.200s", i ? "," : "", 100.0 * top[i].second / total, top[i].first.c_str());
                        text += item;
                    }
                    char tail[64];
                    std::snprintf(tail, sizeof(tail), " (%u samples)\n", total);
                    text += tail;
                }
                if (!text.empty())
                {
                    std::lock_guard<std::mutex> lock(s_hitchMutex);
                    std::fprintf(stderr, "[hitch]  threads from %.3f to %.3f:\n%s", hitchSeconds(request.beginNs), hitchSeconds(request.endNs), text.c_str());
                }
            }

            if (now - windowStart >= std::chrono::seconds(10))
            {
                const double seconds = std::chrono::duration<double>(now - windowStart).count();
                windowStart = now;
                char path[64];
                std::snprintf(path, sizeof(path), "profile_%03d.txt", fileIndex++ % 200);
                if (FILE *f = std::fopen(path, "w"))
                {
                    std::fprintf(f, "window %.1f s (sampling every ~1 ms). Per thread, top functions by samples.\n\n", seconds);
                    std::vector<std::pair<DWORD, ThreadInfo *>> order;
                    for (auto &[tid, info] : threads)
                        if (info.samples)
                            order.push_back({tid, &info});
                    std::sort(order.begin(), order.end(), [](const auto &a, const auto &b)
                              { return a.second->samples > b.second->samples; });
                    for (auto &[tid, info] : order)
                    {
                        std::unordered_map<std::string, uint64_t> byName;
                        for (auto &[rip, count] : info->hits)
                            byName[symbolFor(process, rip, names)] += count;
                        std::vector<std::pair<std::string, uint64_t>> top(byName.begin(), byName.end());
                        std::sort(top.begin(), top.end(), [](const auto &a, const auto &b)
                                  { return a.second > b.second; });
                        std::fprintf(f, "thread %lu: %llu samples (%s)\n", static_cast<unsigned long>(tid),
                                     static_cast<unsigned long long>(info->samples), threadName(info->handle, tid).c_str());
                        for (size_t i = 0; i < top.size() && i < 80; ++i)
                            std::fprintf(f, "  %6.2f%%  %s\n", 100.0 * static_cast<double>(top[i].second) / static_cast<double>(info->samples),
                                         top[i].first.c_str());
                        std::fprintf(f, "\n");
                        info->hits.clear();
                        info->samples = 0;
                    }
                    std::fclose(f);
                }
            }
            Sleep(1);
        }
    }
}

void ps2ProfilerStartIfEnabled()
{
    const char *v = std::getenv("PS2_PROFILE");
    if (!v || !*v || *v == '0')
        return;
    std::thread(samplerMain).detach();
    s_samplerOn.store(true, std::memory_order_relaxed);
    std::fprintf(stderr, "[profile] sampling profiler on: profile_NNN.txt every 10 s\n");
}

#else

void ps2ProfilerStartIfEnabled()
{
}

#endif
