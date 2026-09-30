#pragma once

#include "runtime/gs/gs_backend.h"

#include <memory>

// Vulkan GS backend: draws on the GPU (one render target per GS frame/Z buffer, textures decoded
// from GS memory and cached), everything else (transfers, CLUT, readbacks, presentation) on the
// CPU backend's code with GS memory kept coherent by downloading/uploading render targets as
// needed. It creates its own Vulkan instance and device. Returns nullptr when no suitable device
// is available (Vulkan 1.3 with dynamic rendering and dual-source blending).
std::unique_ptr<GSRasterBackend> ps2CreateVulkanGsBackend();
