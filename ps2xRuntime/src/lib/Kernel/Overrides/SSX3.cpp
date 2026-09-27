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
#include "game_overrides.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

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

    void applySsx3Overrides(PS2Runtime &runtime)
    {
        runtime.replaceFunction(kSifStopModule, ssx3SifStopModule);
        runtime.replaceFunction(kSifUnloadModule, ssx3SifUnloadModule);
        runtime.replaceFunction(kSifSearchModuleByName, ssx3SifSearchModuleByName);
        runtime.replaceFunction(kSifSearchModuleByAddress, ssx3SifSearchModuleByAddress);
        runtime.replaceFunction(kSifSendCmd, ssx3SifSendCmd);
        std::fprintf(stderr, "[ssx3:override] loadfile helpers + SIF SET_SREG mirror installed\n");
    }
}

PS2_REGISTER_GAME_OVERRIDE("SSX 3 loadfile helpers", "SLUS_207.72", 0x00100008u, 0u, applySsx3Overrides);

// ps2_runtime is a static library: without a reference from an always-linked object the
// linker would drop this translation unit and its auto-registration. game_overrides.cpp
// calls this (empty) function to pull it in.
void ps2xForceLinkSsx3Overrides()
{
}
