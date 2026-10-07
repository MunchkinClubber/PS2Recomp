#include "iop_spu2.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <vector>

// The voices are generated in emulated time: one stereo frame per 768 IOP cycles (36.864 MHz /
// 48 kHz), brought up to date whenever the IOP touches the SPU2 and whenever its clock moves on.
// A register write therefore takes effect at the sample it was made for. (They used to be
// rendered by the host's audio thread, in real time, from whatever the registers held at that
// moment - but the IOP runs in bursts: it gets most of a frame's emulated time at once, in a few
// microseconds, and then nothing for 10 ms. A key-on followed by volume steps 4 ms apart, a
// pitch slide, a key-off 20 ms later: all at the same instant for the audio thread. Looping
// sounds whose pitch and volume follow the game came out in steps, short ones not at all.)
// The frames go to the host a millisecond or two's worth at a time (setSpu2VoiceSink), which
// queues them for its audio thread the way it does with the auto-DMA stream.
namespace ps2x::iop::detail::spu2
{
    namespace
    {
        constexpr uint32_t kRamWords = 1u << 20; // 2 MB of 16-bit words
        constexpr uint64_t kCyclesPerFrame = 768u;
        constexpr uint32_t kMaxPending = 16384u; // frames generated in one go at most (after a long pause only the end matters)
        constexpr uint32_t kDeliverFrames = 64u; // handed to the host from this many on

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

        // A volume register with a sweep mode (voice VOLL/VOLR, core MVOLL/MVOLR).
        //  bit 15 clear: bits 14-0 are half the volume, signed (-0x4000..0x3FFF).
        //  bit 15 set: the volume moves by itself from where it is - bit 14 exponential, bit 13
        //  downwards (to 0; else up to the maximum), bit 12 inverted phase, bits 6-0 the rate.
        struct Volume
        {
            uint16_t reg = 0;
            int32_t level = 0; // -0x8000..0x7FFF
            bool sweeping = false, negative = false;
            Envelope env;

            void write(uint16_t value)
            {
                reg = value;
                if (!(value & 0x8000u))
                {
                    sweeping = false;
                    level = static_cast<int16_t>(static_cast<uint16_t>(value << 1));
                    return;
                }
                sweeping = true;
                negative = (value & 0x1000u) != 0u;
                env.reset(static_cast<uint8_t>(value & 0x7Fu), 0x7F, (value & 0x2000u) != 0u, (value & 0x4000u) != 0u);
            }

            void tick()
            {
                if (!sweeping)
                    return;
                const int32_t magnitude = env.tick(std::min(level < 0 ? -level : level, 0x7FFF));
                level = negative ? -magnitude : magnitude;
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
            Volume volL, volR;
            uint16_t pitch = 0, adsr1 = 0, adsr2 = 0;
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
            int16_t before = 0; // the last sample of the block before this one
            int32_t out = 0;    // its last output, envelope applied (what modulates the next voice's pitch)
        };

        struct Core
        {
            std::array<Voice, 24> voices{};
            uint32_t tsa = 0;
            uint32_t endx = 0;
            uint32_t vmixl = 0, vmixr = 0;   // dry output switches
            uint32_t vmixel = 0, vmixer = 0; // effect (reverb) send switches - no reverb here
            uint32_t pmon = 0, non = 0;      // pitch modulation / noise switches
            Volume mvoll, mvolr;
            int16_t avoll = 0x7FFF, avolr = 0x7FFF; // volume of the external input (core 1: core 0's output)
            int16_t bvoll = 0x7FFF, bvolr = 0x7FFF; // volume of the sound data input (the auto-DMA stream)
            uint16_t mmix = 0xFFFFu;                // which sources reach the output
            uint16_t attr = 0;
            uint32_t noiseCount = 0, noiseValue = 1;
            uint16_t regs[0x400 / 2]{};
            // Reverb: its output volume, the ten coefficients (0x774..0x786: IIR, COMB1-4, WALL,
            // APF1, APF2, IN_L, IN_R) - the addresses are in regs (0x2E0..0x33C) - and its state.
            int16_t evoll = 0, evolr = 0;
            int16_t rcoef[10]{};
            uint32_t rpos = 0;      // where the work area's moving origin is (words from ESA)
            uint32_t rcycle = 0;    // frames (left is worked on even ones, right on odd ones)
            int16_t rin[2][64]{};   // what was sent to it, 48 kHz
            int16_t rout[2][32]{};  // what came out, 24 kHz
            uint32_t rcount[2]{};   // samples in rout
            uint8_t rareaSet = 0;   // bit 0: its start address was written, bit 1: its end address
        };

        struct Stats
        {
            uint64_t frames = 0, keyOns = 0, keyOffs = 0, voiceFrames = 0;
            uint32_t peakVoices = 0;
            uint64_t noiseFrames = 0, pitchModFrames = 0, sweepWrites = 0, wetOnlyFrames = 0;
            uint64_t wetFrames = 0, invertedWrites = 0, volumeReads = 0, ended = 0;
            uint64_t sumSquares = 0, reverbSquares = 0;
            uint32_t peak = 0;
            uint64_t delivered = 0, deliveries = 0;
            uint32_t largestDelivery = 0;
        };

        struct State
        {
            // (the IOP thread does everything now; the lock only matters for the statistics and
            // for a host that still calls from elsewhere)
            std::mutex mutex;
            std::vector<uint16_t> ram = std::vector<uint16_t>(kRamWords);
            std::array<Core, 2> cores{};
            const uint64_t *clock = nullptr; // the IOP's cycle counter
            uint64_t framesDone = 0;         // frames generated since the clock was at 0
            std::vector<int16_t> pending; // generated, not yet handed to the host (interleaved stereo)
            Stats stats;
            uint64_t mixerSeen = ~0ull;
            uint32_t mixerLogs = 0;
        };

        State &state()
        {
            static State s;
            return s;
        }

        using VoiceSink = void (*)(const int16_t *stereo, uint32_t frames);
        std::atomic<VoiceSink> s_voiceSink{nullptr};

        const bool s_enabled = []
        {
            const char *v = std::getenv("PS2_SPU2");
            return !(v && v[0] == '0');
        }();
        // PS2_SPU2_BALANCE=0: voices at the level their own volumes give, whatever the stream's
        // input volume is (see mix).
        const bool s_balance = []
        {
            const char *v = std::getenv("PS2_SPU2_BALANCE");
            return !(v && v[0] == '0');
        }();
        const bool s_log = []
        {
            const char *v = std::getenv("PS2_SPU2_LOG");
            return !(v && v[0] == '0');
        }();
        // PS2_SFX_GAIN: the voices' level, in percent (100; 25..400).
        const int32_t s_gain = []
        {
            const char *v = std::getenv("PS2_SFX_GAIN");
            const int percent = v && v[0] ? std::clamp(std::atoi(v), 25, 400) : 100;
            return static_cast<int32_t>(percent * 4096 / 100);
        }();

        void startAttack(Voice &v)
        {
            v.phase = Phase::Attack;
            v.level = 0;
            v.env.reset(static_cast<uint8_t>((v.adsr1 >> 8) & 0x7Fu), 0x7F, false, (v.adsr1 & 0x8000u) != 0u);
        }

        void keyOn(State &s, Core &core, uint32_t index)
        {
            Voice &v = core.voices[index];
            v.nax = v.ssa;
            v.blockValid = false;
            v.position = 0;
            v.s1 = v.s2 = 0;
            v.before = 0;
            v.out = 0;
            v.lsaSetBySoftware = false;
            startAttack(v);
            core.endx &= ~(1u << index);
            ++s.stats.keyOns;
        }

        void keyOff(State &s, Voice &v)
        {
            if (v.phase == Phase::Off || v.phase == Phase::Release)
                return;
            v.phase = Phase::Release;
            v.env.reset(static_cast<uint8_t>((v.adsr2 & 0x1Fu) << 2), 0x1F << 2, true, (v.adsr2 & 0x20u) != 0u);
            ++s.stats.keyOffs;
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
            // Silence after silence (EA's driver parks every free voice in a loop of it): no arithmetic.
            if (v.s1 == 0 && v.s2 == 0)
            {
                uint16_t any = 0;
                for (uint32_t i = 1; i < 8u; ++i)
                    any |= s.ram[(addr + i) & (kRamWords - 1u)];
                if (any == 0u)
                {
                    v.block.fill(0);
                    v.blockValid = true;
                    return;
                }
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

        uint64_t endedBySample = 0; // (statistics: voices silenced by the end of their sample)

        // Advances to the next block after the current one was fully played.
        void finishBlock(Core &core, uint32_t index, Voice &v)
        {
            const uint32_t addr = v.nax & ~7u;
            v.before = v.block[27];
            if (v.blockFlags & 0x01u) // end: on to the loop address, with or without sound
            {
                core.endx |= 1u << index;
                v.nax = v.lsa;
                if (!(v.blockFlags & 0x02u) && v.phase != Phase::Off) // no repeat: silenced
                {
                    v.phase = Phase::Off;
                    v.level = 0;
                    ++endedBySample;
                }
            }
            else
                v.nax = addr + 8u;
            v.nax &= kRamWords - 1u;
            v.blockValid = false;
        }

        // A voice that is off still runs through its data like one that sounds (only silent): its
        // address goes on to the end of the sample and from there to the loop address. Drivers
        // watch that address (EA's: "the voice is free when it is back in my silent loop").
        void idleAdvance(State &s, Core &core, uint32_t index, Voice &v)
        {
            v.out = 0;
            v.position += std::min<uint32_t>(v.pitch, 0x3FFFu);
            while ((v.position >> 12) >= 28u)
            {
                v.position -= 28u << 12;
                if (!v.blockValid) // (only the flags of the block it is in are needed)
                {
                    const uint32_t addr = v.nax & (kRamWords - 1u) & ~7u;
                    v.blockFlags = static_cast<uint8_t>(s.ram[addr] >> 8);
                    if ((v.blockFlags & 0x04u) && !v.lsaSetBySoftware)
                        v.lsa = addr;
                }
                finishBlock(core, index, v);
            }
        }

        // The noise generator of a core (one step per frame; ATTR bits 13-8 are its clock).
        void tickNoise(Core &core)
        {
            static constexpr uint8_t kWaveAdd[64] = {1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0,
                                                     0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1};
            static constexpr uint8_t kFreqAdd[5] = {0, 84, 140, 180, 210};
            const uint32_t clock = (core.attr >> 8) & 0x3Fu;
            const uint32_t level = (0x8000u >> (clock >> 2)) << 16;
            core.noiseCount += 0x10000u + kFreqAdd[clock & 3u];
            if ((core.noiseCount & 0xFFFFu) >= kFreqAdd[4])
            {
                core.noiseCount += 0x10000u;
                core.noiseCount -= kFreqAdd[clock & 3u];
            }
            if (core.noiseCount >= level)
            {
                while (core.noiseCount >= level)
                    core.noiseCount -= level;
                core.noiseValue = (core.noiseValue << 1) | kWaveAdd[(core.noiseValue >> 10) & 63u];
            }
        }

        // One output sample of a voice (envelope applied, before its left/right volumes).
        int32_t renderVoiceSample(State &s, Core &core, uint32_t index, Voice &v)
        {
            if (!v.blockValid)
                decodeBlock(s, v);
            const uint32_t sampleIndex = std::min<uint32_t>(v.position >> 12, 27u);
            // Between the sample before and this one (linear; the real SPU uses a 4-tap gaussian).
            // "The sample before" of a block's first sample is the last one of the block before:
            // interpolating towards the next sample instead, and holding the last sample of every
            // block for want of one, put a step into the sound every 28 samples.
            const int32_t a = sampleIndex != 0u ? v.block[sampleIndex - 1u] : v.before;
            const int32_t b = v.block[sampleIndex];
            const int32_t frac = static_cast<int32_t>(v.position & 0xFFFu);
            int32_t sample = a + (((b - a) * frac) >> 12);
            if (core.non & (1u << index))
                sample = static_cast<int16_t>(core.noiseValue);

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

            // Advance: by the pitch, or - pitch modulation - by the pitch scaled with the output
            // of the voice before (0 .. 2 times).
            uint32_t step = v.pitch;
            if (index != 0u && (core.pmon & (1u << index)))
            {
                const int32_t factor = std::clamp(core.voices[index - 1u].out, -0x8000, 0x7FFF) + 0x8000;
                step = static_cast<uint32_t>((static_cast<int32_t>(static_cast<int16_t>(v.pitch)) * factor) >> 15) & 0xFFFFu;
            }
            step = std::min<uint32_t>(step, 0x3FFFu);
            v.position += step;
            while ((v.position >> 12) >= 28u)
            {
                v.position -= 28u << 12;
                finishBlock(core, index, v);
                decodeBlock(s, v);
            }
            v.out = (sample * v.level) >> 15;
            return v.out;
        }

        inline int32_t scaled(int32_t value, int32_t volume) { return (value * volume) >> 15; }

        // The level of the voices next to the auto-DMA stream, which the host plays as it is.
        // In the machine the stream enters core 0 through the "sound data input" volume BVOL,
        // core 0's whole output enters core 1 through its "external input" volume AVOL, and
        // each core's voices go through its master volume. So against the stream, core 0's
        // voices are at MVOL0 / BVOL0 and core 1's at MVOL1 / (BVOL0 * AVOL1) - of which the
        // master volumes are applied in the mix, and this is the rest: 1 over the path of the
        // stream, as a 4.12 number (between a half and 2; 1 when a volume on the path is not set).
        int32_t balance(int16_t bvol0, int16_t avol1, bool viaCore1)
        {
            if (!s_balance)
                return 1 << 12;
            // (core 0's voices and the stream take the same way through core 1; core 1's do not)
            double path = bvol0 > 0 ? bvol0 / 32767.0 : 1.0;
            if (viaCore1 && avol1 > 0)
                path *= avol1 / 32767.0;
            return static_cast<int32_t>(std::clamp(1.0 / std::max(path, 1e-3), 0.5, 2.0) * 4096.0);
        }

        void logMixer(State &s)
        {
            const Core &c0 = s.cores[0], &c1 = s.cores[1];
            // (what the balance depends on; a line when it changes, a few dozen at most)
            const uint64_t seen = (static_cast<uint64_t>(c0.mvoll.reg) << 48) ^ (static_cast<uint64_t>(c1.mvoll.reg) << 40) ^ (static_cast<uint64_t>(static_cast<uint16_t>(c0.bvoll)) << 24) ^
                                  (static_cast<uint64_t>(static_cast<uint16_t>(c1.avoll)) << 8) ^ (static_cast<uint64_t>(c0.mmix) << 4) ^ c1.mmix ^
                                  (static_cast<uint64_t>(c0.mvolr.reg) << 33) ^ (static_cast<uint64_t>(c1.mvolr.reg) << 17);
            if (seen == s.mixerSeen || s.mixerLogs >= 40u || !s_log)
                return;
            s.mixerSeen = seen;
            ++s.mixerLogs;
            std::fprintf(stderr, "[spu2] mixer: core 0 MVOL %04x/%04x BVOL %04x/%04x AVOL %04x/%04x MMIX %04x | core 1 MVOL %04x/%04x BVOL %04x/%04x AVOL %04x/%04x MMIX %04x | voices against the stream: core 0 x%.2f, core 1 x%.2f\n",
                         c0.mvoll.reg, c0.mvolr.reg, static_cast<uint16_t>(c0.bvoll), static_cast<uint16_t>(c0.bvolr), static_cast<uint16_t>(c0.avoll), static_cast<uint16_t>(c0.avolr), c0.mmix,
                         c1.mvoll.reg, c1.mvolr.reg, static_cast<uint16_t>(c1.bvoll), static_cast<uint16_t>(c1.bvolr), static_cast<uint16_t>(c1.avoll), static_cast<uint16_t>(c1.avolr), c1.mmix,
                         balance(c0.bvoll, c1.avoll, false) / 4096.0, balance(c0.bvoll, c1.avoll, true) / 4096.0);
        }

        const bool s_reverb = []
        {
            const char *v = std::getenv("PS2_SPU2_REVERB");
            return !(v && v[0] == '0');
        }();

        inline int32_t clamp16(int32_t v) { return std::clamp(v, -32768, 32767); }

        // An address register pair of the reverb (high word first), in words.
        inline uint32_t reverbAddress(const Core &c, uint32_t reg)
        {
            return ((static_cast<uint32_t>(c.regs[reg >> 1]) << 16) | c.regs[(reg + 2u) >> 1]) & 0xFFFFFu;
        }

        // The work area of a core's reverb: first word and number of words (0: none).
        inline uint32_t reverbArea(const Core &c, uint32_t &esa)
        {
            esa = ((static_cast<uint32_t>(c.regs[0x2E0 >> 1]) & 0xFu) << 16) | c.regs[0x2E2 >> 1];
            const uint32_t eea = ((static_cast<uint32_t>(c.regs[0x33C >> 1]) & 0xFu) << 16) | 0xFFFFu;
            if (c.rareaSet != 3u) // (not before the driver said where it is: it writes into sound memory)
                return 0u;
            return eea >= esa ? eea - esa + 1u : 0u;
        }

        // One frame of a core's reverb: `inL`/`inR` is what is sent to it, the result is what it
        // adds to the core's output (its own volume EVOL applied).
        //
        // The SPU2 runs the PS1's reverb - reflections through an IIR filter, four combs, two
        // all-pass filters, all of them taps into a work area in sound memory that moves on by
        // one word per frame - on one channel per frame, left and right in turn. Each channel is
        // therefore worked on at 24 kHz: what goes in is low-passed and every second frame taken,
        // what comes out is brought back up to 48 kHz (the same half-band filter both ways).
        void reverbFrame(State &s, Core &c, int32_t inL, int32_t inR, int32_t &outL, int32_t &outR)
        {
            static constexpr int32_t kHalfBand[10] = {10246, -2960, 1332, -616, 266, -103, 35, -10, 2, -1};
            outL = outR = 0;
            uint32_t esa = 0;
            const uint32_t size = reverbArea(c, esa);
            const uint32_t cycle = c.rcycle++;
            c.rin[0][cycle & 63u] = static_cast<int16_t>(clamp16(inL));
            c.rin[1][cycle & 63u] = static_cast<int16_t>(clamp16(inR));
            if (size < 64u || !s_reverb) // (mode "off" is an area of 8 words and all-zero coefficients)
                return;
            if (c.rpos >= size)
                c.rpos = 0;
            const uint32_t ch = cycle & 1u;
            const auto mul = [](int32_t a, int32_t b)
            { return (a * b) >> 15; };

            // In: the frame 19 back, low-passed.
            const int16_t *in = c.rin[ch];
            int32_t acc = 0x4000 * in[(cycle - 19u) & 63u];
            for (uint32_t k = 0; k < 10u; ++k)
                acc += kHalfBand[k] * (in[(cycle - 19u - (2u * k + 1u)) & 63u] + in[(cycle - 19u + (2u * k + 1u)) & 63u]);
            const int32_t x = mul(c.rcoef[8u + ch], clamp16(acc >> 15));

            const auto at = [&](int64_t offset) -> uint16_t &
            {
                // (the presets' offsets lie inside the area: no division on the usual way)
                if (offset < 0 || offset >= static_cast<int64_t>(size))
                {
                    offset %= static_cast<int64_t>(size);
                    if (offset < 0)
                        offset += size;
                }
                uint32_t m = c.rpos + static_cast<uint32_t>(offset);
                if (m >= size)
                    m -= size;
                return s.ram[(esa + m) & (kRamWords - 1u)];
            };
            const auto rd = [&](int64_t offset)
            { return static_cast<int32_t>(static_cast<int16_t>(at(offset))); };
            const bool right = ch != 0u;
            const int64_t sameDst = reverbAddress(c, right ? 0x2F0u : 0x2ECu), diffDst = reverbAddress(c, right ? 0x310u : 0x30Cu);
            const int64_t apf1Dst = reverbAddress(c, right ? 0x330u : 0x32Cu), apf2Dst = reverbAddress(c, right ? 0x338u : 0x334u);
            const int32_t iir = c.rcoef[0], wall = c.rcoef[5], apf1Vol = c.rcoef[6], apf2Vol = c.rcoef[7];
            // Reflections: same side, and from the other side.
            const int32_t samePrev = rd(sameDst - 1), diffPrev = rd(diffDst - 1);
            const int32_t same = mul(iir, x + mul(wall, rd(reverbAddress(c, right ? 0x308u : 0x304u))) - samePrev) + samePrev;
            const int32_t diff = mul(iir, x + mul(wall, rd(reverbAddress(c, right ? 0x324u : 0x328u))) - diffPrev) + diffPrev;
            // Early echo: four combs.
            int32_t out = mul(c.rcoef[1], rd(reverbAddress(c, right ? 0x2F8u : 0x2F4u))) + mul(c.rcoef[2], rd(reverbAddress(c, right ? 0x300u : 0x2FCu))) +
                          mul(c.rcoef[3], rd(reverbAddress(c, right ? 0x318u : 0x314u))) + mul(c.rcoef[4], rd(reverbAddress(c, right ? 0x320u : 0x31Cu)));
            // Late reverb: two all-pass filters.
            const int32_t apf1Src = rd(apf1Dst - static_cast<int64_t>(reverbAddress(c, 0x2E4u)));
            const int32_t apf1 = out - mul(apf1Vol, apf1Src);
            out = apf1Src + mul(apf1Vol, apf1);
            const int32_t apf2Src = rd(apf2Dst - static_cast<int64_t>(reverbAddress(c, 0x2E8u)));
            const int32_t apf2 = out - mul(apf2Vol, apf2Src);
            out = apf2Src + mul(apf2Vol, apf2);
            if (c.attr & 0x80u) // (effect enable: without it the area is read, not written)
            {
                at(sameDst) = static_cast<uint16_t>(clamp16(same));
                at(diffDst) = static_cast<uint16_t>(clamp16(diff));
                at(apf1Dst) = static_cast<uint16_t>(clamp16(apf1));
                at(apf2Dst) = static_cast<uint16_t>(clamp16(apf2));
            }
            c.rpos = c.rpos + 1u >= size ? 0u : c.rpos + 1u;

            // Out: this channel has a new sample now - the frame between two older ones is due;
            // the other channel puts out one of its samples as it is.
            const uint32_t n = c.rcount[ch]++;
            c.rout[ch][n & 31u] = static_cast<int16_t>(clamp16(out));
            acc = 0;
            for (uint32_t k = 0; k < 10u; ++k)
                acc += kHalfBand[k] * (c.rout[ch][(n - 10u - k) & 31u] + c.rout[ch][(n - 9u + k) & 31u]);
            const int32_t between = clamp16(acc >> 14);
            const uint32_t other = ch ^ 1u;
            const int32_t whole = c.rout[other][(c.rcount[other] - 1u - 9u) & 31u];
            outL = mul(ch == 0u ? between : whole, c.evoll);
            outR = mul(ch == 1u ? between : whole, c.evolr);
        }

        // One frame of both cores' voices.
        void mixFrame(State &s, int16_t *out)
        {
            int32_t left = 0, right = 0;
            uint32_t active = 0;
            for (uint32_t ci = 0; ci < 2u; ++ci)
            {
                Core &core = s.cores[ci];
                tickNoise(core);
                int32_t coreL = 0, coreR = 0, wetL = 0, wetR = 0;
                for (uint32_t vi = 0; vi < 24u; ++vi)
                {
                    Voice &v = core.voices[vi];
                    v.volL.tick();
                    v.volR.tick();
                    if (v.phase == Phase::Off)
                    {
                        idleAdvance(s, core, vi, v);
                        continue;
                    }
                    ++active;
                    const uint32_t bit = 1u << vi;
                    const int32_t sample = renderVoiceSample(s, core, vi, v);
                    if (core.non & bit)
                        ++s.stats.noiseFrames;
                    if (vi != 0u && (core.pmon & bit))
                        ++s.stats.pitchModFrames;
                    if (!((core.vmixl | core.vmixr) & bit) && ((core.vmixel | core.vmixer) & bit))
                        ++s.stats.wetOnlyFrames;
                    if ((core.vmixel | core.vmixer) & bit)
                        ++s.stats.wetFrames;
                    const int32_t l = scaled(sample, v.volL.level), r = scaled(sample, v.volR.level);
                    if (core.vmixl & bit)
                        coreL += l;
                    if (core.vmixr & bit)
                        coreR += r;
                    if (core.vmixel & bit)
                        wetL += l;
                    if (core.vmixer & bit)
                        wetR += r;
                }
                core.mvoll.tick();
                core.mvolr.tick();
                // MMIX: which of the voices' sums reach the output (bits 11/10: left/right) and
                // the reverb (bits 9/8). (The other sources it switches - the stream, core 0's
                // output into core 1 - are not mixed here.)
                if (!(core.mmix & 0x800u))
                    coreL = 0;
                if (!(core.mmix & 0x400u))
                    coreR = 0;
                if (!(core.mmix & 0x200u))
                    wetL = 0;
                if (!(core.mmix & 0x100u))
                    wetR = 0;
                int32_t reverbL = 0, reverbR = 0;
                reverbFrame(s, core, wetL, wetR, reverbL, reverbR);
                s.stats.reverbSquares += static_cast<uint64_t>(static_cast<int64_t>(reverbL) * reverbL + static_cast<int64_t>(reverbR) * reverbR) >> 1;
                coreL = scaled(std::clamp(coreL + reverbL, -0x8000 * 4, 0x7FFF * 4), core.mvoll.level);
                coreR = scaled(std::clamp(coreR + reverbR, -0x8000 * 4, 0x7FFF * 4), core.mvolr.level);
                const Core &c0 = s.cores[0], &c1 = s.cores[1];
                left += (coreL * balance(c0.bvoll, c1.avoll, ci == 1u)) >> 12;
                right += (coreR * balance(c0.bvolr, c1.avolr, ci == 1u)) >> 12;
            }
            left = (left * s_gain) >> 12;
            right = (right * s_gain) >> 12;
            out[0] = static_cast<int16_t>(std::clamp(left, -32768, 32767));
            out[1] = static_cast<int16_t>(std::clamp(right, -32768, 32767));
            s.stats.sumSquares += static_cast<uint64_t>(static_cast<int64_t>(out[0]) * out[0] + static_cast<int64_t>(out[1]) * out[1]) >> 1;
            s.stats.peak = std::max<uint32_t>(s.stats.peak, static_cast<uint32_t>(std::max(std::abs(static_cast<int32_t>(out[0])), std::abs(static_cast<int32_t>(out[1])))));
            s.stats.voiceFrames += active;
            s.stats.peakVoices = std::max(s.stats.peakVoices, active);
        }

        void report(State &s)
        {
            Stats &st = s.stats;
            if (s_log && (st.keyOns != 0u || st.voiceFrames != 0u))
            {
                const double frames = static_cast<double>(std::max<uint64_t>(st.frames, 1u));
                std::fprintf(stderr, "[spu2] last 10 s of game time: %llu key-ons, %llu key-offs, %llu samples ran out, %.1f voices sounding on average (%u at most); level rms %.0f peak %u; frames with noise voices %llu, pitch-modulated %llu; volume ramps set %llu, inverted volumes %llu, volume read-backs %llu; %llu frames to the host in %llu pieces (the largest %u)\n",
                             static_cast<unsigned long long>(st.keyOns), static_cast<unsigned long long>(st.keyOffs), static_cast<unsigned long long>(endedBySample - st.ended),
                             static_cast<double>(st.voiceFrames) / frames, st.peakVoices, std::sqrt(static_cast<double>(st.sumSquares) / frames), st.peak,
                             static_cast<unsigned long long>(st.noiseFrames), static_cast<unsigned long long>(st.pitchModFrames),
                             static_cast<unsigned long long>(st.sweepWrites), static_cast<unsigned long long>(st.invertedWrites), static_cast<unsigned long long>(st.volumeReads),
                             static_cast<unsigned long long>(st.delivered), static_cast<unsigned long long>(st.deliveries), st.largestDelivery);
                uint32_t esa0 = 0, esa1 = 0;
                const uint32_t size0 = reverbArea(s.cores[0], esa0), size1 = reverbArea(s.cores[1], esa1);
                if (st.wetFrames != 0u || st.reverbSquares != 0u || size0 >= 64u || size1 >= 64u)
                    std::fprintf(stderr, "[spu2] reverb%s: core 0 %s, area %u words, volume %04x/%04x | core 1 %s, area %u words, volume %04x/%04x | %.1f voices sent to it on average (%.1f of them only to it), its output rms %.0f\n",
                                 s_reverb ? "" : " (switched off: PS2_SPU2_REVERB=0)",
                                 (s.cores[0].attr & 0x80u) ? "on" : "off", size0, static_cast<uint16_t>(s.cores[0].evoll), static_cast<uint16_t>(s.cores[0].evolr),
                                 (s.cores[1].attr & 0x80u) ? "on" : "off", size1, static_cast<uint16_t>(s.cores[1].evoll), static_cast<uint16_t>(s.cores[1].evolr),
                                 static_cast<double>(st.wetFrames) / frames, static_cast<double>(st.wetOnlyFrames) / frames, std::sqrt(static_cast<double>(st.reverbSquares) / (2.0 * frames)));
            }
            const uint64_t ended = endedBySample;
            st = Stats{};
            st.ended = ended;
        }

        // Generates the frames up to the IOP's clock and hands them to the host (the lock is held).
        void advance(State &s)
        {
            if (!s.clock || !s_enabled)
                return;
            const uint64_t target = *s.clock / kCyclesPerFrame;
            if (target < s.framesDone) // (the clock was reset)
                s.framesDone = target;
            if (target - s.framesDone > kMaxPending)
                s.framesDone = target - kMaxPending;
            while (s.framesDone < target)
            {
                const size_t at = s.pending.size();
                s.pending.resize(at + 2u);
                mixFrame(s, s.pending.data() + at);
                ++s.framesDone;
                if (++s.stats.frames >= 480000u)
                    report(s);
            }
            const uint32_t frames = static_cast<uint32_t>(s.pending.size() / 2u);
            if (frames < kDeliverFrames)
                return;
            if (const VoiceSink sink = s_voiceSink.load(std::memory_order_relaxed))
                sink(s.pending.data(), frames);
            s.stats.delivered += frames;
            ++s.stats.deliveries;
            s.stats.largestDelivery = std::max(s.stats.largestDelivery, frames);
            s.pending.clear();
        }

        uint32_t *voiceMaskFor(Core &core, uint32_t reg)
        {
            switch (reg & ~2u)
            {
            case 0x180: return &core.pmon;
            case 0x184: return &core.non;
            case 0x188: return &core.vmixl;
            case 0x18C: return &core.vmixel;
            case 0x190: return &core.vmixr;
            case 0x194: return &core.vmixer;
            default: return nullptr;
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
        s.framesDone = 0;
        s.pending.clear();
        s.stats = Stats{};
    }

    void bindClock(const uint64_t *cycles)
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.clock = cycles;
        s.framesDone = cycles ? *cycles / kCyclesPerFrame : 0u;
    }

    void sync()
    {
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        advance(s);
    }

    void write16(uint32_t phys, uint16_t value)
    {
        if (!s_enabled || phys < kBase || phys >= kEnd)
            return;
        State &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        advance(s); // (what was before this write is generated with the old value)
        const uint32_t off = phys - kBase;
        // 0x760.. holds the per-core volumes (core 1 at +0x28).
        if (off >= 0x760u && off < 0x7B0u)
        {
            const uint32_t coreIndex = off >= 0x788u ? 1u : 0u;
            const uint32_t r = off - 0x760u - coreIndex * 0x28u;
            Core &c = s.cores[coreIndex];
            switch (r)
            {
            case 0x00:
                c.mvoll.write(value);
                s.stats.sweepWrites += value >> 15;
                break;
            case 0x02:
                c.mvolr.write(value);
                s.stats.sweepWrites += value >> 15;
                break;
            case 0x04: c.evoll = static_cast<int16_t>(value); return;
            case 0x06: c.evolr = static_cast<int16_t>(value); return;
            case 0x08: c.avoll = static_cast<int16_t>(value); break;
            case 0x0A: c.avolr = static_cast<int16_t>(value); break;
            case 0x0C: c.bvoll = static_cast<int16_t>(value); break;
            case 0x0E: c.bvolr = static_cast<int16_t>(value); break;
            default:
                if (r >= 0x14u && r < 0x28u) // the reverb's coefficients
                    c.rcoef[(r - 0x14u) >> 1] = static_cast<int16_t>(value);
                return;
            }
            logMixer(s);
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
            case 0x0:
            case 0x2:
                ((reg & 0xFu) == 0u ? v.volL : v.volR).write(value);
                s.stats.sweepWrites += value >> 15;
                s.stats.invertedWrites += (value & 0x8000u) ? (value >> 12) & 1u : (value >> 14) & 1u;
                break;
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
        case 0x2E0:
        case 0x2E2:
            c.rpos = 0; // (a new work area starts at its beginning)
            c.rareaSet |= 1u;
            break;
        case 0x33C:
            c.rareaSet |= 2u;
            break;
        case 0x198:
            c.mmix = value;
            logMixer(s);
            break;
        case 0x19A: c.attr = value; break;
        case 0x1A0:
        case 0x1A2:
        {
            const uint32_t first = reg == 0x1A0u ? 0u : 16u;
            for (uint32_t i = 0; i < 16u && first + i < 24u; ++i)
                if (value & (1u << i))
                    keyOn(s, c, first + i);
            break;
        }
        case 0x1A4:
        case 0x1A6:
        {
            const uint32_t first = reg == 0x1A4u ? 0u : 16u;
            for (uint32_t i = 0; i < 16u && first + i < 24u; ++i)
                if (value & (1u << i))
                    keyOff(s, c.voices[first + i]);
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
        if (reg < 0x180u && (reg & 0xFu) >= 0xAu)
        {
            advance(s);
            const Voice &v = c.voices[reg >> 4];
            switch (reg & 0xFu)
            {
            case 0xA: value = static_cast<uint16_t>(v.level); return true;
            // The volume as it is at this moment: the whole signed 16-bit level (a fixed
            // register value reads back doubled). EA's driver steers its volume ramps by it.
            case 0xC: value = static_cast<uint16_t>(v.volL.level); ++s.stats.volumeReads; return true;
            case 0xE: value = static_cast<uint16_t>(v.volR.level); ++s.stats.volumeReads; return true;
            default: return false;
            }
        }
        if (reg >= 0x1C0u && reg < 0x1C0u + 24u * 0xCu)
        {
            const uint32_t field = (reg - 0x1C0u) % 0xCu;
            if (field != 0x8u && field != 0xAu)
                return false;
            advance(s);
            const Voice &v = c.voices[(reg - 0x1C0u) / 0xCu];
            value = field == 0x8u ? static_cast<uint16_t>(v.nax >> 16) : static_cast<uint16_t>(v.nax);
            return true;
        }
        if (reg == 0x340u || reg == 0x342u)
        {
            advance(s);
            value = reg == 0x340u ? static_cast<uint16_t>(c.endx) : static_cast<uint16_t>(c.endx >> 16);
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
        advance(s);
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
    // Declared at its use in the host (ps2_runtime.cpp).
    void setSpu2VoiceSink(void (*sink)(const int16_t *stereo, uint32_t frames));

    // Receives the voices' output as it is generated: 48 kHz interleaved stereo, 48000 frames per
    // second of emulated time, in pieces from about a millisecond up to as much as the IOP's
    // clock moved at once. Called on the IOP's thread; must not call back into the SPU2.
    void setSpu2VoiceSink(void (*sink)(const int16_t *stereo, uint32_t frames))
    {
        detail::spu2::s_voiceSink.store(sink, std::memory_order_relaxed);
    }

    // (The voices used to be rendered here, by the host's audio thread.)
    void spu2Render(int16_t *stereo, uint32_t frames)
    {
        (void)stereo;
        (void)frames;
    }
}
