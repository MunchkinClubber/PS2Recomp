#pragma once

#include <cstdint>

// Stutter diagnostics (ps2_profiler.cpp): what took long when a frame came late. Included only by
// the few files that report something (not part of what the recompiled game code includes).
//
// In the log:
//   [slow] t=123.456 GS thread: waited for the render thread to take the frame: 31.2 ms
//   [hitch] t=123.470 game frame took 48.1 ms (...)
//   [hitch]   GameThread: 61% FUN_0037e238, 22% NtReadFile, ... (24 samples)
// t is seconds since the program started, so the lines of one hitch can be put side by side.
// The per-thread lines need the sampling profiler (PS2_PROFILE=1): they are its samples from the
// time the frame took.

// Steady clock, nanoseconds (the same clock the GS code uses).
uint64_t ps2HitchNowNs();

// Something on a frame's way took `ns`. One line (at most 40 in two seconds).
void ps2SlowLog(const char *thread, const char *what, uint64_t ns);

// A game frame or a frame's pictures took too long, from beginNs to endNs: one line, and what
// every thread was executing in that time. `what` is the whole text of the line. At most one
// report every 150 ms.
void ps2HitchReport(const char *what, uint64_t beginNs, uint64_t endNs);

// The name the profiler shows for the calling thread (ps2_memory.cpp).
void ps2NameThisThread(const char *name);

// From here on a wait is worth a [slow] line.
constexpr uint64_t kSlowNs = 8000000ull;
