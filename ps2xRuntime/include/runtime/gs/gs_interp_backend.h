#pragma once

#include "runtime/gs/gs_backend.h"

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
// With factor 1 it only forwards (no buffering, no cost). Calls that need the renderer's state
// right now (read-backs from the EE, GS memory reads) execute what is buffered first; such a
// frame is shown without in-between pictures.
std::unique_ptr<GSRasterBackend> ps2CreateInterpGsBackend(std::unique_ptr<GSRasterBackend> inner);

// Pictures per game frame (1 = off, up to 8). Takes effect at the next flip. PS2_FRAME_INTERP
// sets the start value.
void ps2GsInterpSetFactor(uint32_t factor);
uint32_t ps2GsInterpFactor();

// GS thread, in command order: the primitives that follow come from this "object" (a VU1 program
// call, or a GIF packet sent directly). `hash` identifies its input data.
void ps2GsInterpObjectTag(uint32_t pc, uint32_t hash);

// GS thread, in command order: a local->host read whose data is not needed right now. True: it
// was queued and `done` is called with the data when the real frame is drawn. False: not
// buffering, read it directly.
bool ps2GsInterpDeferReadback(uint32_t bytes, std::function<void(std::vector<uint8_t> &&data, uint32_t got)> done);

// When the next picture queued with QueuePresentSnapshot should be shown (steady_clock ns since
// its epoch; 0 = as soon as it is ready). Read by the renderer that shows the pictures.
uint64_t ps2GsInterpNextPictureDueNs();

// Tests: called after each pass's flip with the renderer that drew it.
extern void (*g_gsInterpPassHook)(GSRasterBackend *inner, uint32_t pass, uint32_t passes, float t);
