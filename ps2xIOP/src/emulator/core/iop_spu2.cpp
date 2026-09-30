#include "iop_spu2.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace ps2x::iop::detail::spu2
{
    namespace
    {
        constexpr uint32_t kRamWords = 1u << 20; // 2 MB of 16-bit words

        // ADSR / volume envelope (the PS1/PS2 SPU envelope generator).
        struct Envelope
        {
            int32_t counter = 0, counterInc = 0, step = 0;
            uint8_t rate = 0;
            bool decreasing = false, exponential = false;

            void reset(uint8_t newRate, uint8_t rateMask, bool dec, bool exp)
            {
                rate = newRate;
                decreasing = dec;
                exponential = exp;
                counter = 0;
                counterInc = 0x8000;
                const int32_t base = 7 - static_cast<int32_t>(rate & 3u);
                step = dec ? ~base : base;
                if (rate < 44u)
                    step <<= (11 - (rate >> 2));
                else if (rate >= 48u)
                    counterInc >>= ((rate >> 2) - 11);
                if ((rate & rateMask) == rateMask)
                    counterInc = 0;
            }

            int32_t tick(int32_t level)
            {
                int32_t thisStep = step;
                int32_t thisInc = counterInc;
                if (exponential)
                {
                    if (decreasing)
                        thisStep = (thisStep * level) >> 15;
                    else if (level >= 0x6000)
                    {
                        if (rate < 40u)
                            thisStep >>= 2;
                        else if (rate >= 44u)
                            thisInc >>= 2;
                        else
                        {
                            thisStep >>= 1;
                            thisInc >>= 1;
                        }
                    }
                }
                counter += thisInc;
                if (!(counter & 0x8000))
                    return level;
                counter = 0;
                return std::clamp(level + thisStep, 0, 0x7FFF);
            }
        };

        enum class Phase : uint8_t
        {
            Off,
            Attack,
            Decay,
            Sustain,
            Release
        };

        struct Voice
        {
            uint16_t volL = 0, volR = 0, pitch = 0, adsr1 = 0, adsr2 = 0;
            uint32_t ssa = 0, lsa = 0, nax = 0; // in 16-bit words
            bool lsaSetBySoftware = false;
            Phase phase = Phase::Off;
            int32_t level = 0;
            Envelope env;
            // ADPCM decoding
            std::array<int16_t, 28> block{};
            int32_t s1 = 0, s2 = 0;
            uint32_t position = 0; // 12-bit fraction within the decoded block (sample = position >> 12)
            bool blockValid = false;
            uint8_t blockFlags = 0;
        };

        struct Core
        {
            std::array<Voice, 24> voices{};
            uint32_t tsa = 0;
            uint32_t endx = 0;
            uint32_t vmixl = 0, vmixr = 0;
            uint16_t mvoll = 0x3FFF, mvolr = 0x3FFF;
            uint16_t regs[0x400 / 2]{};
        };

        struct State
        {
            std::mutex mutex;
            std::vector<uint16_t> ram = std::vector<uint16_t>(kRamWords);
            std::array<Core, 2> cores{};
            uint64_t keyOns = 0;
        };

        State &state()
        {
            static State s;
            return s;
        }

        const bool s_enabled = []
        {
            const char *v = std::getenv("PS2_SPU2");
            return !(v && v[0] == '0');
        }();

        float volumeFactor(uint16_t v)
        {
            if (v & 0x8000u)
                return 0.5f; // sweep mode: not simulated, use a middle level
            return static_cast<float>(static_cast<int16_t>(static_cast<uint16_t>(v << 1))) / 32768.0f;
        }

        void startAttack(Voice &v)
        {
            v.phase = Phase::Attack;
            v.level = 0;
            v.env.reset(static_cast<uint8_t>((v.adsr1 >> 8) & 0x7Fu), 0x7F, false, (v.adsr1 & 0x8000u) != 0u);
        }

        void keyOn(Core &core, uint32_t index)
        {
            Voice &v = core.voices[index];
            v.nax = v.ssa;
            v.blockValid = false;
            v.position = 0;
            v.s1 = v.s2 = 0;
            v.lsaSetBySoftware = false;
            startAttack(v);
            core.endx &= ~(1u << index);
            ++state().keyOns;
        }

        void keyOff(Voice &v)
        {
            if (v.phase == Phase::Off || v.phase == Phase::Release)
                return;
            v.phase = Phase::Release;
            v.env.reset(static_cast<uint8_t>((v.adsr2 & 0x1Fu) << 2), 0x1F << 2, true, (v.adsr2 & 0x20u) != 0u);
        }

        void decodeBlock(State &s, Voice &v)
        {
            static constexpr int32_t kF0[5] = {0, 60, 115, 98, 122};
            static constexpr int32_t kF1[5] = {0, 0, -52, -55, -60};
            const uint32_t addr = v.nax & (kRamWords - 1u) & ~7u;
            const uint16_t header = s.ram[addr];
            const uint32_t shift = header & 0x0Fu;
            const uint32_t filter = std::min<uint32_t>((header >> 4) & 0x07u, 4u);
            v.blockFlags = static_cast<uint8_t>(header >> 8);
            if (v.blockFlags & 0x04u) // loop start
            {
                if (!v.lsaSetBySoftware)
                    v.lsa = addr;
            }
            for (uint32_t i = 0; i < 28u; ++i)
            {
                const uint16_t word = s.ram[(addr + 1u + (i >> 2)) & (kRamWords - 1u)];
                const int32_t nibble = static_cast<int16_t>(static_cast<uint16_t>(((word >> ((i & 3u) * 4u)) & 0x0Fu) << 12));
                int32_t sample = nibble >> (shift > 12u ? 9u : shift); // shifts 13-15 act as 9
                sample += (v.s1 * kF0[filter] + v.s2 * kF1[filter] + 32) >> 6;
                sample = std::clamp(sample, -32768, 32767);
                v.block[i] = static_cast<int16_t>(sample);
                v.s2 = v.s1;
                v.s1 = sample;
            }
            v.blockValid = true;
        }

        // Advances to the next block after the current one was fully played.
        void finishBlock(Core &core, uint32_t index, Voice &v)
        {
            const uint32_t addr = v.nax & ~7u;
            if (v.blockFlags & 0x01u) // end
            {
                core.endx |= 1u << index;
                if (v.blockFlags & 0x02u) // repeat: continue at the loop address
                    v.nax = v.lsa;
                else
                {
                    v.phase = Phase::Off;
                    v.level = 0;
                    v.nax = addr + 8u;
                }
            }
            else
                v.nax = addr + 8u;
            v.nax &= kRamWords - 1u;
            v.blockValid = false;
        }

        int32_t renderVoiceSample(State &s, Core &core, uint32_t index, Voice &v)
        {
            if (v.phase == Phase::Off)
                return 0;
            if (!v.blockValid)
                decodeBlock(s, v);
            const uint32_t sampleIndex = v.position >> 12;
            // Linear interpolation between neighbouring decoded samples (the real SPU uses a
            // 4-tap gaussian; this is close enough for playback).
            const int32_t a = v.block[std::min<uint32_t>(sampleIndex, 27u)];
            const int32_t b = sampleIndex + 1u < 28u ? v.block[sampleIndex + 1u] : a;
            const int32_t frac = static_cast<int32_t>(v.position & 0xFFFu);
            const int32_t sample = a + (((b - a) * frac) >> 12);

            // Envelope
            switch (v.phase)
            {
            case Phase::Attack:
                v.level = v.env.tick(v.level);
                if (v.level >= 0x7FFF)
                {
                    v.phase = Phase::Decay;
                    v.env.reset(static_cast<uint8_t>(((v.adsr1 >> 4) & 0x0Fu) << 2), 0x1F << 2, true, true);
                }
                break;
            case Phase::Decay:
            {
                v.level = v.env.tick(v.level);
                const int32_t sustainLevel = std::min<int32_t>(((v.adsr1 & 0x0Fu) + 1) * 0x800, 0x7FFF);
                if (v.level <= sustainLevel)
                {
                    v.phase = Phase::Sustain;
                    v.env.reset(static_cast<uint8_t>((v.adsr2 >> 6) & 0x7Fu), 0x7F, (v.adsr2 & 0x4000u) != 0u, (v.adsr2 & 0x8000u) != 0u);
                }
                break;
            }
            case Phase::Sustain:
                v.level = v.env.tick(v.level);
                break;
            case Phase::Release:
                v.level = v.env.tick(v.level);
                if (v.level <= 0)
                {
                    v.phase = Phase::Off;
                    v.level = 0;
                }
                break;
            default:
                break;
            }

            // Advance
            const uint32_t step = std::min<uint32_t>(v.pitch, 0x3FFFu);
            v.position += step;
            while ((v.position >> 12) >= 28u && v.phase != Phase::Off)
            {
                v.position -= 28u << 12;
                finishBlock(core, index, v);
                if (v.phase != Phase::Off)
                    decodeBlock(s, v);
            }
            return (sample * v.level) >> 15;
        }

        uint32_t *voiceMaskFor(Core &core, uint32_t reg)
        {
            switch (reg)
            {
            case 0x188:
            case 0x18A:
                return &core.vmixl;
            case 0x190:
            case 0x192:
                return &core.vmixr;
            default:
                return nullptr;
            }
        }
    }

    void reset()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        std::fill(s.ram.begin(), s.ram.end(), uint16_t{0});
        for (Core &c : s.cores)
            c = Core{};
    }

    void write16(uint32_t phys, uint16_t value)
    {
        if (!s_enabled || phys < kBase || phys >= kEnd)
            return;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const uint32_t off = phys - kBase;
        // 0x760.. holds the per-core master volumes (core 1 at +0x28).
        if (off >= 0x760u && off < 0x7B0u)
        {
            const uint32_t coreIndex = off >= 0x788u ? 1u : 0u;
            const uint32_t r = off - 0x760u - coreIndex * 0x28u;
            Core &c = s.cores[coreIndex];
            if (r == 0u)
                c.mvoll = value;
            else if (r == 2u)
                c.mvolr = value;
            return;
        }
        const uint32_t coreIndex = (off >> 10) & 1u;
        const uint32_t reg = off & 0x3FFu;
        Core &c = s.cores[coreIndex];
        c.regs[reg >> 1] = value;

        if (reg < 0x180u)
        {
            Voice &v = c.voices[reg >> 4];
            switch (reg & 0xFu)
            {
            case 0x0: v.volL = value; break;
            case 0x2: v.volR = value; break;
            case 0x4: v.pitch = value; break;
            case 0x6: v.adsr1 = value; break;
            case 0x8: v.adsr2 = value; break;
            case 0xA: v.level = static_cast<int32_t>(value & 0x7FFFu); break;
            default: break;
            }
            return;
        }
        if (reg >= 0x1C0u && reg < 0x1C0u + 24u * 0xCu)
        {
            Voice &v = c.voices[(reg - 0x1C0u) / 0xCu];
            const uint32_t field = (reg - 0x1C0u) % 0xCu;
            auto setHi = [&](uint32_t &a)
            { a = (a & 0xFFFFu) | ((static_cast<uint32_t>(value) & 0xFu) << 16); };
            auto setLo = [&](uint32_t &a)
            { a = (a & 0xF0000u) | value; };
            switch (field)
            {
            case 0x0: setHi(v.ssa); break;
            case 0x2: setLo(v.ssa); break;
            case 0x4: setHi(v.lsa); v.lsaSetBySoftware = true; break;
            case 0x6: setLo(v.lsa); v.lsaSetBySoftware = true; break;
            case 0x8: setHi(v.nax); break;
            case 0xA: setLo(v.nax); break;
            default: break;
            }
            return;
        }
        if (uint32_t *mask = voiceMaskFor(c, reg))
        {
            if ((reg & 2u) == 0u)
                *mask = (*mask & 0xFF0000u) | value;
            else
                *mask = (*mask & 0xFFFFu) | ((static_cast<uint32_t>(value) & 0xFFu) << 16);
            return;
        }
        switch (reg)
        {
        case 0x1A0:
        case 0x1A2:
        {
            const uint32_t first = reg == 0x1A0u ? 0u : 16u;
            for (uint32_t i = 0; i < 16u && first + i < 24u; ++i)
                if (value & (1u << i))
                    keyOn(c, first + i);
            break;
        }
        case 0x1A4:
        case 0x1A6:
        {
            const uint32_t first = reg == 0x1A4u ? 0u : 16u;
            for (uint32_t i = 0; i < 16u && first + i < 24u; ++i)
                if (value & (1u << i))
                    keyOff(c.voices[first + i]);
            break;
        }
        case 0x1A8: c.tsa = (c.tsa & 0xFFFFu) | ((static_cast<uint32_t>(value) & 0xFu) << 16); break;
        case 0x1AA: c.tsa = (c.tsa & 0xF0000u) | value; break;
        case 0x1AC:
            s.ram[c.tsa & (kRamWords - 1u)] = value;
            c.tsa = (c.tsa + 1u) & (kRamWords - 1u);
            break;
        case 0x340: c.endx &= ~static_cast<uint32_t>(value); break; // write clears (not used by all drivers)
        default: break;
        }
    }

    bool read16(uint32_t phys, uint16_t &value)
    {
        if (!s_enabled || phys < kBase || phys >= kEnd)
            return false;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        const uint32_t off = phys - kBase;
        if (off >= 0x760u)
            return false;
        const uint32_t coreIndex = (off >> 10) & 1u;
        const uint32_t reg = off & 0x3FFu;
        Core &c = s.cores[coreIndex];
        if (reg < 0x180u && (reg & 0xFu) == 0xAu)
        {
            value = static_cast<uint16_t>(c.voices[reg >> 4].level);
            return true;
        }
        if (reg >= 0x1C0u && reg < 0x1C0u + 24u * 0xCu)
        {
            const uint32_t field = (reg - 0x1C0u) % 0xCu;
            const Voice &v = c.voices[(reg - 0x1C0u) / 0xCu];
            if (field == 0x8u)
            {
                value = static_cast<uint16_t>(v.nax >> 16);
                return true;
            }
            if (field == 0xAu)
            {
                value = static_cast<uint16_t>(v.nax);
                return true;
            }
            return false;
        }
        if (reg == 0x340u)
        {
            value = static_cast<uint16_t>(c.endx);
            return true;
        }
        if (reg == 0x342u)
        {
            value = static_cast<uint16_t>(c.endx >> 16);
            return true;
        }
        if (reg == 0x1A8u)
        {
            value = static_cast<uint16_t>(c.tsa >> 16);
            return true;
        }
        if (reg == 0x1AAu)
        {
            value = static_cast<uint16_t>(c.tsa);
            return true;
        }
        return false;
    }

    void dma(uint32_t core, uint8_t *iopRam, uint32_t bytes, bool toSpu)
    {
        if (!s_enabled || !iopRam)
            return;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        Core &c = s.cores[core & 1u];
        const uint32_t words = bytes / 2u;
        for (uint32_t i = 0; i < words; ++i)
        {
            uint16_t *w = &s.ram[c.tsa & (kRamWords - 1u)];
            if (toSpu)
                std::memcpy(w, iopRam + i * 2u, 2u);
            else
                std::memcpy(iopRam + i * 2u, w, 2u);
            c.tsa = (c.tsa + 1u) & (kRamWords - 1u);
        }
    }
}

namespace ps2x::iop
{
    void spu2Render(int16_t *stereo, uint32_t frames)
    {
        using namespace detail::spu2;
        if (!s_enabled)
            return;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        static uint64_t s_reported = 0;
        if (s.keyOns != 0u && s_reported == 0u)
        {
            s_reported = s.keyOns;
            std::fprintf(stderr, "[spu2] first voice key-on seen: core0 vmix %06x/%06x mvol %04x/%04x, core1 vmix %06x/%06x mvol %04x/%04x\n",
                         s.cores[0].vmixl, s.cores[0].vmixr, s.cores[0].mvoll, s.cores[0].mvolr,
                         s.cores[1].vmixl, s.cores[1].vmixr, s.cores[1].mvoll, s.cores[1].mvolr);
        }
        for (uint32_t f = 0; f < frames; ++f)
        {
            float left = 0.0f, right = 0.0f;
            for (uint32_t ci = 0; ci < 2u; ++ci)
            {
                Core &core = s.cores[ci];
                float coreL = 0.0f, coreR = 0.0f;
                for (uint32_t vi = 0; vi < 24u; ++vi)
                {
                    Voice &v = core.voices[vi];
                    if (v.phase == Phase::Off)
                        continue;
                    const float sample = static_cast<float>(renderVoiceSample(s, core, vi, v));
                    if (core.vmixl & (1u << vi))
                        coreL += sample * volumeFactor(v.volL);
                    if (core.vmixr & (1u << vi))
                        coreR += sample * volumeFactor(v.volR);
                }
                left += coreL * volumeFactor(core.mvoll);
                right += coreR * volumeFactor(core.mvolr);
            }
            const int32_t outL = std::clamp(static_cast<int32_t>(stereo[2u * f]) + static_cast<int32_t>(left), -32768, 32767);
            const int32_t outR = std::clamp(static_cast<int32_t>(stereo[2u * f + 1u]) + static_cast<int32_t>(right), -32768, 32767);
            stereo[2u * f] = static_cast<int16_t>(outL);
            stereo[2u * f + 1u] = static_cast<int16_t>(outR);
        }
    }
}
