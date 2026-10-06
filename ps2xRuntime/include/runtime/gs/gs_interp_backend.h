#pragma once

#include "runtime/gs/gs_backend.h"

#include <bitset>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// Frame interpolation ("more pictures than the game draws"). The game still runs at its own rate
// (one frame per vertical blank); this layer sits between the GS front end and a renderer, keeps
// each game frame's renderer calls until the flip, matches its primitives with the previous
// frame's (same VU1 program + same model data + same position in the program's output), and then
// draws the frame several times: first with the matched vertices part of the way from the
// previous frame's positions, last as the game sent it. Every pass is an ordinary frame for the
// renderer and ends with a flip, so the renderer shows factor pictures per game frame.
//
// Primitives without a partner (clipped differently, new, meshes the game rebuilds every frame)
// are placed by a camera model fitted to the matched ones, or carried along with their object or
// surroundings. Texture memory that the frame reads and then replaces is put back between the
// passes, so that every pass draws with the textures the game meant (see findHazards).
//
// With factor 1 it only forwards (no buffering, no cost). Calls that need the renderer's state
// right now (read-backs from the EE, GS memory reads) execute what is buffered first; such a
// frame is shown without in-between pictures.
//
// (This header is only included by the few files that need it: gs_backend.h is part of what all
// the recompiled game code includes, so the calls the layer needs beyond that interface live
// here.)

// What a renderer can offer the layer beyond GSRasterBackend.
class GSRasterBackendEx
{
public:
    virtual ~GSRasterBackendEx() = default;
    // Brings the given 8 KiB pages of GS memory (the buffer passed to Initialize) up to date with
    // everything submitted so far, so the caller can read them there.
    virtual void SyncPages(const std::bitset<512> &pages) = 0;
    // A local->host transfer (direction 1) whose data is not needed right away: instead of
    // BeginTransfer + ConsumeLocalToHostBytes, which wait for everything drawn so far, the
    // renderer reads the data as of this point of the command stream and calls `done` with all
    // of the transfer's bytes when it has them (on whichever thread is in the renderer then;
    // `done` must not call the renderer). False: not now - use the usual calls.
    virtual bool ReadbackAsync(const GSTransferCommand &command, std::function<void(std::vector<uint8_t> &&)> done) = 0;
};

// `ex`: the same renderer's GSRasterBackendEx, if it has one (stays owned by `inner`).
std::unique_ptr<GSRasterBackend> ps2CreateInterpGsBackend(std::unique_ptr<GSRasterBackend> inner, GSRasterBackendEx *ex = nullptr);

// Pictures per game frame (1 = off, up to 8). Takes effect at the next flip. PS2_FRAME_INTERP
// sets the start value.
void ps2GsInterpSetFactor(uint32_t factor);
uint32_t ps2GsInterpFactor();

// GS thread, in command order: the primitives that follow come from this "object" (a VU1 program
// call, or a GIF packet sent directly). `hash` identifies its input data.
void ps2GsInterpObjectTag(uint32_t pc, uint32_t hash);

// GS thread, in command order, right after the packet that set up a local->host transfer: its
// data is not needed right now. True: `done` is called with the data later - when the real frame
// is drawn (interpolating) and/or when the renderer has it without having waited for the GPU.
// False: read it directly.
bool ps2GsInterpDeferReadback(uint32_t bytes, std::function<void(std::vector<uint8_t> &&data, uint32_t got)> done);

// When the next picture queued with QueuePresentSnapshot should be shown (steady_clock ns since
// its epoch; 0 = as soon as it is ready). Read by the renderer that shows the pictures.
uint64_t ps2GsInterpNextPictureDueNs();

// Tests: called after each pass's flip with the renderer that drew it.
extern void (*g_gsInterpPassHook)(GSRasterBackend *inner, uint32_t pass, uint32_t passes, float t);
