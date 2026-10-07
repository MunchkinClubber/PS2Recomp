#pragma once

#include <cstdint>

// SPU2 voice emulation for host audio: SPU RAM (manual and DMA transfers), the 2x24 ADPCM voices
// with ADSR envelopes, key on/off, noise, pitch modulation, volume sweeps, ENDX/ENVX/VOLX/NAX
// read-back, voice/master volumes, the mix switches and the reverb. Everything happens on the
// IOP's thread, in emulated time: 48000 frames per second of the IOP's clock (see bindClock),
// handed to the host as they are generated (ps2x::iop::setSpu2VoiceSink in iop_spu2.cpp).
namespace ps2x::iop::detail::spu2
{
    constexpr uint32_t kBase = 0x1F900000u;
    constexpr uint32_t kEnd = 0x1F900800u;

    void reset();
    // The IOP's cycle counter (36.864 MHz): what the voices are generated against.
    void bindClock(const uint64_t *cycles);
    // Generates the frames up to the clock's present value (also done by every access below).
    void sync();
    // A 16-bit register write in [kBase, kEnd).
    void write16(uint32_t phys, uint16_t value);
    // Registers whose value the SPU2 itself changes (ENVX, NAX, ENDX): true and the value.
    bool read16(uint32_t phys, uint16_t &value);
    // Non-auto DMA between IOP RAM and SPU RAM at the core's TSA (toSpu: IOP RAM -> SPU RAM).
    void dma(uint32_t core, uint8_t *iopRam, uint32_t bytes, bool toSpu);
}
