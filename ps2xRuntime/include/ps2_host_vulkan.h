#pragma once
// The SDL3 host's Vulkan device, shared with the Vulkan GS renderer so that the frames it draws
// are shown straight from the GPU (no copy through the CPU). Include only where volk is available.

#include <volk.h>

#include <cstdint>
#include <mutex>

struct HostVulkanShared
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE; // the one queue: every vkQueueSubmit / vkQueuePresentKHR /
    uint32_t queueFamily = 0;       // vkDeviceWaitIdle holds *queueMutex
    std::mutex *queueMutex = nullptr;
    bool depthClamp = false;
    bool dualSrcBlend = false;
    // Timeline semaphore the host signals after each present that read a renderer frame (the
    // value passed to the frame provider), so the renderer knows when it may overwrite an image.
    VkSemaphore releaseSemaphore = VK_NULL_HANDLE;
};

// nullptr unless the host's Vulkan is up on a Vulkan 1.3 device with dynamic rendering and
// timeline semaphores. Retain/Release keep the device alive past CloseWindow while in use.
const HostVulkanShared *HostVulkanGetShared();
void HostVulkanRetain();
void HostVulkanRelease();

// A frame drawn on the shared device: `image` (RGBA8) is in TRANSFER_SRC_OPTIMAL layout once the
// timeline semaphore `ready` reaches `readyValue`; the picture is its top-left width x height.
struct HostGpuFrame
{
    VkImage image = VK_NULL_HANDLE;
    uint32_t imageWidth = 0, imageHeight = 0;
    uint32_t width = 0, height = 0;
    VkSemaphore ready = VK_NULL_HANDLE;
    uint64_t readyValue = 0;
};

// Called by the host on its present (main thread, holding the queue mutex): return true and fill
// `out` to show that frame instead of the texture drawn with DrawTexturePro. The host will signal
// releaseSemaphore to `releaseValue` once it has finished reading the image.
using HostGpuFrameProvider = bool (*)(void *user, HostGpuFrame &out, uint64_t releaseValue);
void HostVulkanSetFrameProvider(HostGpuFrameProvider provider, void *user);
