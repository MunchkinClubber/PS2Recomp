// Game override: SSX 3 (NTSC-U, SLUS_207.72)
//
// The game calls several libkernel "loadfile" helpers that are not replaced by the
// recompiler's stub bindings: sceSifStopModule, sceSifUnloadModule,
// sceSifSearchModuleByName and sceSifSearchModuleByAddress. Each of them first runs
// _lf_init (0x42B068), which binds the IOP LOADFILE RPC server (0x80000006). The HLE
// runtime has no such server, so _lf_init spins forever in its retry/delay loop and
// the worker thread that loads the game's data never makes progress (stuck on the
// loading screen). Route these entry points to host-side implementations instead.

#include "../Syscalls/Common.h"
#include "../Syscalls/Sync.h"
#include "game_overrides.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

void ps2xReservePrivateGuestHeap(PS2Runtime &runtime, uint32_t base, uint32_t limit); // ps2_runtime.cpp

namespace
{
    constexpr int32_t kKeUnknownModule = -202;

    constexpr uint32_t kSifStopModule = 0x0042B438u;
    constexpr uint32_t kSifUnloadModule = 0x0042B640u;
    constexpr uint32_t kSifSearchModuleByName = 0x0042B6D0u;
    constexpr uint32_t kSifSearchModuleByAddress = 0x0042B770u;

    std::string lowerAscii(std::string s)
    {
        for (char &c : s)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return s;
    }

    // "cdrom0:\DATA\MODULES\USBD.IRX;1" -> "usbd"
    std::string moduleStemFromPath(const std::string &path)
    {
        std::string p = path;
        const size_t semi = p.find(';');
        if (semi != std::string::npos)
        {
            p.resize(semi);
        }
        const size_t slash = p.find_last_of("\\/:");
        if (slash != std::string::npos)
        {
            p = p.substr(slash + 1);
        }
        const size_t dot = p.find('.');
        if (dot != std::string::npos)
        {
            p.resize(dot);
        }
        return lowerAscii(p);
    }

    std::string readGuestString(uint8_t *rdram, uint32_t address, size_t maxLen = 128)
    {
        std::string out;
        if (address == 0u)
        {
            return out;
        }
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char c = static_cast<char>(rdram[(address + i) & PS2_RAM_MASK]);
            if (c == '\0')
            {
                break;
            }
            out.push_back(c);
        }
        return out;
    }

    void returnToCaller(R5900Context *ctx)
    {
        ctx->pc = getRegU32(ctx, 31);
    }

    void ssx3SifStopModule(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        returnToCaller(ctx);
        ps2_syscalls::SifStopModule(rdram, ctx, runtime);
        std::fprintf(stderr, "[ssx3:lf] sceSifStopModule(id=%d) -> %d\n", (int)getRegU32(ctx, 4), (int)getRegU32(ctx, 2));
    }

    void ssx3SifUnloadModule(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        returnToCaller(ctx);
        const int32_t id = static_cast<int32_t>(getRegU32(ctx, 4));
        int32_t result = kKeUnknownModule;
        {
            std::lock_guard<std::mutex> lock(g_sif_module_mutex);
            auto it = g_sif_modules_by_id.find(id);
            if (it != g_sif_modules_by_id.end())
            {
                it->second.loaded = false;
                result = id;
            }
        }
        setReturnS32(ctx, result);
        std::fprintf(stderr, "[ssx3:lf] sceSifUnloadModule(id=%d) -> %d\n", (int)id, (int)result);
    }

    void ssx3SifSearchModuleByName(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        returnToCaller(ctx);
        const std::string wanted = readGuestString(rdram, getRegU32(ctx, 4));
        const std::string wantedLower = lowerAscii(wanted);
        int32_t result = kKeUnknownModule;
        std::string matchedPath;
        {
            std::lock_guard<std::mutex> lock(g_sif_module_mutex);
            for (const auto &[id, record] : g_sif_modules_by_id)
            {
                if (!record.loaded || wantedLower.empty())
                {
                    continue;
                }
                const std::string stem = moduleStemFromPath(record.path);
                if (stem.empty())
                {
                    continue;
                }
                // IRX internal names ("USB_driver", "dev9_driver", ...) usually contain the file stem.
                if (wantedLower == stem || wantedLower.find(stem) != std::string::npos || stem.find(wantedLower) != std::string::npos)
                {
                    result = id;
                    matchedPath = record.path;
                    break;
                }
            }
        }
        setReturnS32(ctx, result);
        std::fprintf(stderr, "[ssx3:lf] sceSifSearchModuleByName(\"%s\") -> %d %s\n", wanted.c_str(), (int)result, matchedPath.c_str());
    }

    void ssx3SifSearchModuleByAddress(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        (void)runtime;
        returnToCaller(ctx);
        setReturnS32(ctx, kKeUnknownModule);
        std::fprintf(stderr, "[ssx3:lf] sceSifSearchModuleByAddress(0x%x) -> %d\n", (unsigned)getRegU32(ctx, 4), (int)kKeUnknownModule);
    }

    // EE-side libkernel SIF software-register table (sceSifGetSreg 0x425CF0 / sceSifSetSreg 0x425D08).
    constexpr uint32_t kEeSifSregTable = 0x0052BE00u;
    constexpr uint32_t kSifSendCmd = 0x004261B0u;
    constexpr uint32_t kSifCmdSetSreg = 0x80000001u;

    // SSX 3 boot handshake (FUN_0040B130): the EE sends SIF_CMD_SET_SREG(1, 1) to the IOP and then
    // spins until its own sreg 1 becomes non-zero, i.e. until an IOP module echoes readiness back.
    // The HLE runtime drops SIF commands, so nothing ever answers. Mirror SET_SREG into the EE-side
    // table (EE and IOP are one machine here), then continue with the normal handler.
    void ssx3SifSendCmd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        returnToCaller(ctx);
        const uint32_t cid = getRegU32(ctx, 4);
        const uint32_t packet = getRegU32(ctx, 5);
        if (cid == kSifCmdSetSreg && packet != 0u)
        {
            uint32_t reg = 0u;
            uint32_t value = 0u;
            std::memcpy(&reg, rdram + ((packet + 0x10u) & PS2_RAM_MASK), sizeof(reg));
            std::memcpy(&value, rdram + ((packet + 0x14u) & PS2_RAM_MASK), sizeof(value));
            if (reg < 32u)
            {
                std::memcpy(rdram + kEeSifSregTable + reg * 4u, &value, sizeof(value));
                std::fprintf(stderr, "[ssx3:sif] SET_SREG(%u, 0x%x) mirrored to EE sreg table\n", (unsigned)reg, (unsigned)value);
            }
        }
        ps2_syscalls::sceSifSendCmd(rdram, ctx, runtime);
    }

    // Renderer DMA state machine (object base in a0, state word at +0x5a8c):
    //   0 idle -> kick GIF(+0x5ab0) & VIF1(+0x5aa8), state 1
    //   VIF1 done (0x382688): 1 -> 2, iSignalSema(+0x5ac8)
    //   T5 in state 2: kick GIF(+0x5aa0) & VIF1(+0x5a9c), state 3
    //   VIF1 done (0x382688): 3 -> 4
    //   GIF done (0x3825f8): ++count(+0x5abc); in state 4 with count >= limit(+0x5ab8) -> 5, signal
    //   T5 in state 5: flip, state 0 (main thread busy-waits on state==0 at 0x382730).
    // On hardware the PATH3 GIF chain finishes after VIF1 (PATH1/2 have priority), so the GIF
    // interrupt lands in state 4. Here both DMAs complete instantly and the GIF completion is
    // delivered first, so state 4 is never left unless some unrelated GIF DMA happens to follow
    // (the FMV's image uploads did, which is why the movie ran and the game froze right after).
    // Treat GIF completions that already happened as arriving after VIF1.
    constexpr uint32_t kRendererVif1Done = 0x00382688u;
    // crt0 (0x100148) sets the main stack to 0x1FE0000-0x2000000 and EndOfHeap stays 0x1F00000, so
    // 0x1F00000-0x1FE0000 is unused by the game.
    constexpr uint32_t kSsx3RuntimeHeapBase = 0x01F00000u;
    constexpr uint32_t kSsx3RuntimeHeapLimit = 0x01FE0000u;
    constexpr uint32_t kGifChcr = 0x1000A000u;

    void ssx3RendererVif1Done(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        returnToCaller(ctx);
        const uint32_t obj = getRegU32(ctx, 4);
        auto rd = [&](uint32_t off) -> uint32_t
        {
            uint32_t v = 0u;
            std::memcpy(&v, rdram + ((obj + off) & PS2_RAM_MASK), sizeof(v));
            return v;
        };
        auto wr = [&](uint32_t off, uint32_t v)
        { std::memcpy(rdram + ((obj + off) & PS2_RAM_MASK), &v, sizeof(v)); };

        const uint32_t state = rd(0x5a8cu);
        bool signal = false;
        if (state == 1u)
        {
            wr(0x5a8cu, 2u);
            signal = true;
        }
        else if (state == 3u)
        {
            wr(0x5a8cu, 4u);
            const bool gifIdle = (runtime->memory().readIORegister(kGifChcr) & 0x100u) == 0u;
            const int32_t count = static_cast<int32_t>(rd(0x5abcu));
            const int32_t limit = static_cast<int32_t>(rd(0x5ab8u));
            if (gifIdle && count >= limit)
            {
                wr(0x5a8cu, 5u);
                wr(0x5abcu, 0u);
                signal = true;
            }
            static int logCount = 0;
            if (logCount < 8)
            {
                ++logCount;
                std::fprintf(stderr, "[ssx3:render] VIF1 done in state 3: gifIdle=%d count=%d limit=%d -> state %u\n",
                             gifIdle ? 1 : 0, (int)count, (int)limit, (unsigned)rd(0x5a8cu));
            }
        }
        if (signal)
        {
            const uint32_t ra = getRegU32(ctx, 31);
            SET_GPR_U32(ctx, 4, rd(0x5ac8u));
            ps2_syscalls::iSignalSema(rdram, ctx, runtime);
            ctx->pc = ra;
        }
    }

    void applySsx3Overrides(PS2Runtime &runtime)
    {
        runtime.replaceFunction(kSifStopModule, ssx3SifStopModule);
        runtime.replaceFunction(kSifUnloadModule, ssx3SifUnloadModule);
        runtime.replaceFunction(kSifSearchModuleByName, ssx3SifSearchModuleByName);
        runtime.replaceFunction(kSifSearchModuleByAddress, ssx3SifSearchModuleByAddress);
        runtime.replaceFunction(kSifSendCmd, ssx3SifSendCmd);
        runtime.replaceFunction(kRendererVif1Done, ssx3RendererVif1Done);
        // The game's allocator (init at 0x31AED0) takes [malloc(0x400)+0x800, EndOfHeap()), i.e. all RAM
        // from SetupHeap's base -- exactly where the runtime put its own heap (sceMpegCreate buffers,
        // MPEG callback data, SIF packets...). Those overlapping allocations corrupted the game's free
        // lists, and after the EA intro malloc_consolidate (0x31EEE8) looped forever on a cyclic bin.
        // Give the runtime the unused 896 KB between EndOfHeap and the main stack instead.
        ps2xReservePrivateGuestHeap(runtime, kSsx3RuntimeHeapBase, kSsx3RuntimeHeapLimit);
        std::fprintf(stderr, "[ssx3:override] loadfile helpers + SIF SET_SREG mirror + renderer DMA ordering fix + private runtime heap installed\n");
    }
}

PS2_REGISTER_GAME_OVERRIDE("SSX 3 loadfile helpers", "SLUS_207.72", 0x00100008u, 0u, applySsx3Overrides);

// ps2_runtime is a static library: without a reference from an always-linked object the
// linker would drop this translation unit and its auto-registration. game_overrides.cpp
// calls this (empty) function to pull it in.
void ps2xForceLinkSsx3Overrides()
{
}
