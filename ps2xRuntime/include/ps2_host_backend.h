#pragma once

#if defined(PS2X_HOST_SDL3)
#include "ps2_host_sdl3.h" // SDL3 window/input/audio + Vulkan presentation, raylib-compatible API
#else
#include "raylib.h"
#endif
