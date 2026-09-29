// Built-in sampling profiler (Windows). Set PS2_PROFILE=1: a background thread samples the
// instruction pointer of every thread in the process about once per millisecond and every
// 10 seconds writes profile_NNN.txt with the hottest functions per thread for that window
// (symbols from ps2EntryRunner.pdb via DbgHelp).

#include <cstdio>
#include <cstdlib>
#include <cstring>

void ps2ProfilerStartIfEnabled();

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
            name = symbol->Name;
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

    void samplerMain()
    {
        const DWORD self = GetCurrentThreadId();
        const DWORD pid = GetCurrentProcessId();
        HANDLE process = GetCurrentProcess();
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
        SymInitialize(process, nullptr, TRUE);

        struct ThreadInfo
        {
            HANDLE handle = nullptr;
            std::unordered_map<DWORD64, uint32_t> hits;
            uint64_t samples = 0;
        };
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
                        std::fprintf(f, "thread %lu: %llu samples\n", static_cast<unsigned long>(tid),
                                     static_cast<unsigned long long>(info->samples));
                        for (size_t i = 0; i < top.size() && i < 40; ++i)
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
    std::fprintf(stderr, "[profile] sampling profiler on: profile_NNN.txt every 10 s\n");
}

#else

void ps2ProfilerStartIfEnabled()
{
}

#endif
