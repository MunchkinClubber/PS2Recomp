#pragma once

#include "runtime/gs/gs_backend.h"

#include <memory>

// Vulkan GS backend: draws on the GPU (one render target per GS frame/Z buffer, textures decoded
// from a GPU mirror of GS memory and cached); transfers, CLUT loads and readbacks run on the CPU
// backend's code over GS memory, kept coherent with the mirror. With the SDL3 host it shares the
// window's Vulkan device and builds the displayed picture on the GPU at each flip, which the host
// shows directly (see ps2_host_vulkan.h); otherwise it creates its own device and presents through
// the CPU. Returns nullptr when no suitable device is available (Vulkan 1.3 with dynamic rendering).
//
// Internal resolution: PS2_GS_SCALE=1..8 (render pixels per GS pixel in each direction, capped
// by the GPU's limits; F8 in game cycles 1x-4x). GS memory stays at native resolution; render
// targets keep the extra detail, including a 32-bit buffer reinterpreted as 16-bit and back.
//
// Environment: PS2_GS_VK_CPUPRESENT=1 present through the CPU, PS2_GS_VK_OWNDEVICE=1 do not share
// the window's device, PS2_GS_VK_CHECKPRESENT=1 compare each GPU picture with the CPU one (slow).
std::unique_ptr<GSRasterBackend> ps2CreateVulkanGsBackend();

// The extra calls of a backend made by ps2CreateVulkanGsBackend (for the frame interpolation
// layer, see gs_interp_backend.h).
class GSRasterBackendEx;
GSRasterBackendEx *ps2VulkanGsBackendEx(GSRasterBackend *backend);
