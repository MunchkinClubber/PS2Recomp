#pragma once

#include <cstdint>

// Minimal SPU2 voice emulation for host audio: SPU RAM (manual and DMA transfers), the 2x24
// ADPCM voices with ADSR envelopes, key on/off, ENDX/ENVX/NAX read-back, voice/master volumes and
// the dry mix switches. No reverb, no pitch modulation or noise. The IOP thread drives register
// writes; the host audio thread renders (spu2Render in iop_host.h). Thread-safe.
namespace ps2x::iop::detail::spu2
{
    constexpr uint32_t kBase = 0x1F900000u;
    constexpr uint32_t kEnd = 0x1F900800u;

    void reset();
    // A 16-bit register write in [kBase, kEnd).
    void write16(uint32_t phys, uint16_t value);
    // Registers whose value the SPU2 itself changes (ENVX, NAX, ENDX): true and the value.
    bool read16(uint32_t phys, uint16_t &value);
    // Non-auto DMA between IOP RAM and SPU RAM at the core's TSA (toSpu: IOP RAM -> SPU RAM).
    void dma(uint32_t core, uint8_t *iopRam, uint32_t bytes, bool toSpu);
}
