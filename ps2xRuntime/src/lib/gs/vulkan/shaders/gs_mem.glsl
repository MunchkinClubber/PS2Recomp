// GS local memory on the GPU: a 4 MiB mirror (vram) and the page swizzle tables of
// ps2_gs_memory (lut, 16-bit entries packed in pairs). pixelAddr() is PixelStorageTraits::Address.
layout(std430, set = 0, binding = 0) buffer Vram { uint vram[]; };
layout(std430, set = 0, binding = 1) readonly buffer Lut { uint lut[]; };

uint lutEntry(uint idx)
{
    return (lut[idx >> 1] >> ((idx & 1u) * 16u)) & 0xFFFFu;
}

// Pixel index (in units of the format's unpacked width) of (x, y) in a buffer at block `bp`
// of width `bw` (64-pixel units); `pw` x `ph` is the page size, `tableOff` the table's first entry.
uint pixelAddr(uint tableOff, uint pw, uint ph, uint bp, uint bw, uint x, uint y)
{
    const uint page = bp / 32u + (y / ph) * ((bw * 64u) / pw) + x / pw;
    const uint block = bp % 32u;
    return page * (pw * ph) + lutEntry(tableOff + (block * ph + (y % ph)) * pw + (x % pw));
}
