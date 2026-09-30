// SDL3 + Vulkan implementation of the raylib-compatible host API in ps2_host_sdl3.h.
//
// Window, input and audio come from SDL3. Frames are presented with Vulkan: the runtime's one
// frame texture is copied into a GPU image and blitted (scaled) into the swapchain image. This is
// the foundation the Vulkan renderer builds on; for now it shows exactly what the CPU renderer drew.
//
// Environment switches:
//   PS2_VSYNC=1          present with FIFO (vsync) instead of MAILBOX/IMMEDIATE
//   PS2_FILTER=linear    bilinear scaling of the game image (default: nearest, as before)
//   PS2_VK_VALIDATION=1  enable the Khronos validation layer if installed
//   Alt+Enter            toggle fullscreen

#include "ps2_host_sdl3.h"

#include <volk.h>
#include <SDL3/SDL.h>
#define SDL_MAIN_HANDLED 1 // the runtime has its own main(); only SDL_SetMainReady is wanted
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace
{
    // ---------------------------------------------------------------- state
    SDL_Window *g_window = nullptr;
    bool g_windowReady = false;
    bool g_shouldClose = false;
    unsigned int g_configFlags = 0;
    int g_targetFps = 0;
    uint64_t g_nextFrameNs = 0;

    // Input snapshot (written on the main thread in pumpEvents, read from any thread).
    constexpr int kMaxKeys = 512;
    std::array<std::atomic<uint8_t>, kMaxKeys> g_keyDown{};
    std::array<uint8_t, kMaxKeys> g_keyPrev{};
    std::array<uint8_t, kMaxKeys> g_keyPressed{};
    SDL_Gamepad *g_gamepad = nullptr;
    std::atomic<bool> g_gamepadPresent{false};
    std::atomic<uint32_t> g_padButtons{0};             // bit per raylib GamepadButton
    std::array<std::atomic<int16_t>, 6> g_padAxes{};     // raylib GamepadAxis order

    // Textures (the runtime uses one: the game frame).
    struct HostTexture
    {
        int width = 0, height = 0;
        std::vector<uint8_t> pixels; // RGBA8
        bool dirty = true;
    };
    std::unordered_map<unsigned int, HostTexture> g_textures;
    unsigned int g_nextTextureId = 1;
    struct DrawCall
    {
        unsigned int texture = 0;
        Rectangle src{}, dst{};
        bool valid = false;
    };
    DrawCall g_draw;
    Color g_clearColor{0, 0, 0, 255};

    // ---------------------------------------------------------------- Vulkan
    constexpr uint32_t kFramesInFlight = 2;
    struct FrameResources
    {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        void *stagingPtr = nullptr;
        VkDeviceSize stagingSize = 0;
        VkImage image = VK_NULL_HANDLE; // the game frame on the GPU
        VkDeviceMemory imageMemory = VK_NULL_HANDLE;
        int imageWidth = 0, imageHeight = 0;
    };

    struct Vk
    {
        bool ok = false;
        VkInstance instance = VK_NULL_HANDLE;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkPhysicalDevice physical = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t queueFamily = 0;
        VkQueue queue = VK_NULL_HANDLE;
        VkPhysicalDeviceMemoryProperties memProps{};
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VkFormat swapFormat = VK_FORMAT_UNDEFINED;
        VkExtent2D swapExtent{};
        std::vector<VkImage> swapImages;
        std::vector<VkSemaphore> renderDone; // per swapchain image
        bool swapchainDirty = true;
        FrameResources frames[kFramesInFlight];
        uint32_t frameIndex = 0;
        bool linearFilter = false;
        bool vsync = false;
    } g_vk;

    bool envIs(const char *name, const char *value)
    {
        const char *v = std::getenv(name);
        return v && SDL_strcasecmp(v, value) == 0;
    }

    void logVk(const char *what, VkResult r)
    {
        std::fprintf(stderr, "[host:vk] %s failed (VkResult %d)\n", what, static_cast<int>(r));
    }

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props)
    {
        for (uint32_t i = 0; i < g_vk.memProps.memoryTypeCount; ++i)
            if ((typeBits & (1u << i)) && (g_vk.memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        return UINT32_MAX;
    }

    bool createInstance()
    {
        if (volkInitialize() != VK_SUCCESS)
        {
            std::fprintf(stderr, "[host:vk] Vulkan loader not found (is the graphics driver installed?)\n");
            return false;
        }
        Uint32 extCount = 0;
        const char *const *sdlExts = SDL_Vulkan_GetInstanceExtensions(&extCount);
        std::vector<const char *> exts(sdlExts, sdlExts + extCount);
        std::vector<const char *> layers;
        if (envIs("PS2_VK_VALIDATION", "1"))
            layers.push_back("VK_LAYER_KHRONOS_validation");

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "PS2Recomp";
        app.pEngineName = "ps2xRuntime";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;
        ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ci.ppEnabledExtensionNames = exts.data();
        ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
        ci.ppEnabledLayerNames = layers.data();
        VkResult r = vkCreateInstance(&ci, nullptr, &g_vk.instance);
        if (r != VK_SUCCESS && !layers.empty())
        {
            std::fprintf(stderr, "[host:vk] validation layer unavailable, continuing without it\n");
            ci.enabledLayerCount = 0;
            r = vkCreateInstance(&ci, nullptr, &g_vk.instance);
        }
        if (r != VK_SUCCESS)
        {
            logVk("vkCreateInstance", r);
            return false;
        }
        volkLoadInstance(g_vk.instance);
        return true;
    }

    bool pickDevice()
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(g_vk.instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(g_vk.instance, &count, devices.data());
        int bestScore = -1;
        for (VkPhysicalDevice d : devices)
        {
            uint32_t qCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, nullptr);
            std::vector<VkQueueFamilyProperties> qs(qCount);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qCount, qs.data());
            for (uint32_t q = 0; q < qCount; ++q)
            {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(d, q, g_vk.surface, &present);
                if (!(qs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present)
                    continue;
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(d, &props);
                int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                            : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                                                                                         : 1;
                if (score > bestScore)
                {
                    bestScore = score;
                    g_vk.physical = d;
                    g_vk.queueFamily = q;
                }
                break;
            }
        }
        if (!g_vk.physical)
        {
            std::fprintf(stderr, "[host:vk] no Vulkan device can present to this window\n");
            return false;
        }
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(g_vk.physical, &props);
        vkGetPhysicalDeviceMemoryProperties(g_vk.physical, &g_vk.memProps);
        std::fprintf(stderr, "[host:vk] using %s (Vulkan %u.%u.%u, driver 0x%x)\n", props.deviceName,
                     VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
                     VK_API_VERSION_PATCH(props.apiVersion), props.driverVersion);
        return true;
    }

    bool createDevice()
    {
        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = g_vk.queueFamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        const char *exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = &qci;
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = exts;
        const VkResult r = vkCreateDevice(g_vk.physical, &ci, nullptr, &g_vk.device);
        if (r != VK_SUCCESS)
        {
            logVk("vkCreateDevice", r);
            return false;
        }
        volkLoadDevice(g_vk.device);
        vkGetDeviceQueue(g_vk.device, g_vk.queueFamily, 0, &g_vk.queue);
        return true;
    }

    bool createFrameResources()
    {
        for (FrameResources &f : g_vk.frames)
        {
            VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pci.queueFamilyIndex = g_vk.queueFamily;
            if (vkCreateCommandPool(g_vk.device, &pci, nullptr, &f.pool) != VK_SUCCESS)
                return false;
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = f.pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(g_vk.device, &ai, &f.cmd) != VK_SUCCESS)
                return false;
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            if (vkCreateFence(g_vk.device, &fci, nullptr, &f.fence) != VK_SUCCESS ||
                vkCreateSemaphore(g_vk.device, &sci, nullptr, &f.imageAvailable) != VK_SUCCESS)
                return false;
        }
        return true;
    }

    void destroyFrameImage(FrameResources &f)
    {
        if (f.image)
            vkDestroyImage(g_vk.device, f.image, nullptr);
        if (f.imageMemory)
            vkFreeMemory(g_vk.device, f.imageMemory, nullptr);
        if (f.staging)
            vkDestroyBuffer(g_vk.device, f.staging, nullptr);
        if (f.stagingMemory)
            vkFreeMemory(g_vk.device, f.stagingMemory, nullptr);
        f.image = VK_NULL_HANDLE;
        f.imageMemory = VK_NULL_HANDLE;
        f.staging = VK_NULL_HANDLE;
        f.stagingMemory = VK_NULL_HANDLE;
        f.stagingPtr = nullptr;
        f.stagingSize = 0;
        f.imageWidth = f.imageHeight = 0;
    }

    // Per-frame GPU copy of the game frame (and its upload buffer), sized to the texture.
    bool ensureFrameImage(FrameResources &f, int width, int height)
    {
        if (f.image && f.imageWidth == width && f.imageHeight == height)
            return true;
        destroyFrameImage(f);
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * 4u;

        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(g_vk.device, &bci, nullptr, &f.staging) != VK_SUCCESS)
            return false;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(g_vk.device, f.staging, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g_vk.device, &mai, nullptr, &f.stagingMemory) != VK_SUCCESS)
            return false;
        vkBindBufferMemory(g_vk.device, f.staging, f.stagingMemory, 0);
        if (vkMapMemory(g_vk.device, f.stagingMemory, 0, bytes, 0, &f.stagingPtr) != VK_SUCCESS)
            return false;
        f.stagingSize = bytes;

        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1u};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(g_vk.device, &ici, nullptr, &f.image) != VK_SUCCESS)
            return false;
        vkGetImageMemoryRequirements(g_vk.device, f.image, &req);
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g_vk.device, &mai, nullptr, &f.imageMemory) != VK_SUCCESS)
            return false;
        vkBindImageMemory(g_vk.device, f.image, f.imageMemory, 0);
        f.imageWidth = width;
        f.imageHeight = height;
        return true;
    }

    void destroySwapchain(VkSwapchainKHR swapchain)
    {
        for (VkSemaphore s : g_vk.renderDone)
            vkDestroySemaphore(g_vk.device, s, nullptr);
        g_vk.renderDone.clear();
        g_vk.swapImages.clear();
        if (swapchain)
            vkDestroySwapchainKHR(g_vk.device, swapchain, nullptr);
    }

    bool createSwapchain()
    {
        int pw = 0, ph = 0;
        SDL_GetWindowSizeInPixels(g_window, &pw, &ph);
        if (pw <= 0 || ph <= 0)
            return false; // minimised

        vkDeviceWaitIdle(g_vk.device);
        VkSurfaceCapabilitiesKHR caps{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk.physical, g_vk.surface, &caps);
        if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0)
            return false;

        uint32_t fmtCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.physical, g_vk.surface, &fmtCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(fmtCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.physical, g_vk.surface, &fmtCount, formats.data());
        VkSurfaceFormatKHR chosen = formats.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR} : formats[0];
        for (const VkSurfaceFormatKHR &f : formats)
            if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                chosen = f;
                break;
            }

        uint32_t pmCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.physical, g_vk.surface, &pmCount, nullptr);
        std::vector<VkPresentModeKHR> modes(pmCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.physical, g_vk.surface, &pmCount, modes.data());
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        if (!g_vk.vsync)
        {
            if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end())
                mode = VK_PRESENT_MODE_MAILBOX_KHR;
            else if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end())
                mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }

        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX)
        {
            extent.width = std::clamp(static_cast<uint32_t>(pw), caps.minImageExtent.width, caps.maxImageExtent.width);
            extent.height = std::clamp(static_cast<uint32_t>(ph), caps.minImageExtent.height, caps.maxImageExtent.height);
        }
        uint32_t imageCount = std::max(caps.minImageCount + 1u, 3u);
        if (caps.maxImageCount != 0)
            imageCount = std::min(imageCount, caps.maxImageCount);
        if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        {
            std::fprintf(stderr, "[host:vk] the swapchain cannot be a copy destination\n");
            return false;
        }

        VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        sci.surface = g_vk.surface;
        sci.minImageCount = imageCount;
        sci.imageFormat = chosen.format;
        sci.imageColorSpace = chosen.colorSpace;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = mode;
        sci.clipped = VK_TRUE;
        VkSwapchainKHR old = g_vk.swapchain;
        sci.oldSwapchain = old;
        VkSwapchainKHR created = VK_NULL_HANDLE;
        const VkResult r = vkCreateSwapchainKHR(g_vk.device, &sci, nullptr, &created);
        destroySwapchain(old);
        g_vk.swapchain = VK_NULL_HANDLE;
        if (r != VK_SUCCESS)
        {
            logVk("vkCreateSwapchainKHR", r);
            return false;
        }
        g_vk.swapchain = created;
        g_vk.swapFormat = chosen.format;
        g_vk.swapExtent = extent;

        uint32_t count = 0;
        vkGetSwapchainImagesKHR(g_vk.device, g_vk.swapchain, &count, nullptr);
        g_vk.swapImages.resize(count);
        vkGetSwapchainImagesKHR(g_vk.device, g_vk.swapchain, &count, g_vk.swapImages.data());
        g_vk.renderDone.resize(count);
        VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (VkSemaphore &s : g_vk.renderDone)
            vkCreateSemaphore(g_vk.device, &semci, nullptr, &s);

        // Linear scaling needs the frame format to support filtered blits (it does on every desktop GPU).
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(g_vk.physical, VK_FORMAT_R8G8B8A8_UNORM, &fp);
        g_vk.linearFilter = envIs("PS2_FILTER", "linear") && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);

        static int s_logs = 0;
        if (s_logs++ < 8)
            std::fprintf(stderr, "[host:vk] swapchain %ux%u, %u images, %s\n", extent.width, extent.height, count,
                         mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo (vsync)");
        g_vk.swapchainDirty = false;
        return true;
    }

    bool initVulkan()
    {
        g_vk.vsync = envIs("PS2_VSYNC", "1");
        if (!createInstance())
            return false;
        if (!SDL_Vulkan_CreateSurface(g_window, g_vk.instance, nullptr, &g_vk.surface))
        {
            std::fprintf(stderr, "[host:vk] SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
            return false;
        }
        if (!pickDevice() || !createDevice() || !createFrameResources())
            return false;
        createSwapchain(); // may be deferred if the window starts minimised
        g_vk.ok = true;
        return true;
    }

    void shutdownVulkan()
    {
        if (!g_vk.device)
            return;
        vkDeviceWaitIdle(g_vk.device);
        for (FrameResources &f : g_vk.frames)
        {
            destroyFrameImage(f);
            if (f.fence)
                vkDestroyFence(g_vk.device, f.fence, nullptr);
            if (f.imageAvailable)
                vkDestroySemaphore(g_vk.device, f.imageAvailable, nullptr);
            if (f.pool)
                vkDestroyCommandPool(g_vk.device, f.pool, nullptr);
            f = FrameResources{};
        }
        destroySwapchain(g_vk.swapchain);
        g_vk.swapchain = VK_NULL_HANDLE;
        vkDestroyDevice(g_vk.device, nullptr);
        if (g_vk.surface)
            SDL_Vulkan_DestroySurface(g_vk.instance, g_vk.surface, nullptr);
        if (g_vk.instance)
            vkDestroyInstance(g_vk.instance, nullptr);
        g_vk = Vk{};
    }

    void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                      VkAccessFlags srcAccess, VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;
        b.newLayout = to;
        b.srcAccessMask = srcAccess;
        b.dstAccessMask = dstAccess;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    void presentFrame()
    {
        if (!g_vk.ok)
            return;
        if (g_vk.swapchainDirty || !g_vk.swapchain)
            if (!createSwapchain())
                return; // minimised or not ready yet

        FrameResources &f = g_vk.frames[g_vk.frameIndex];
        vkWaitForFences(g_vk.device, 1, &f.fence, VK_TRUE, UINT64_MAX);

        uint32_t imageIndex = 0;
        VkResult r = vkAcquireNextImageKHR(g_vk.device, g_vk.swapchain, UINT64_MAX, f.imageAvailable, VK_NULL_HANDLE, &imageIndex);
        if (r == VK_ERROR_OUT_OF_DATE_KHR)
        {
            g_vk.swapchainDirty = true;
            return;
        }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR)
        {
            logVk("vkAcquireNextImageKHR", r);
            return;
        }
        vkResetFences(g_vk.device, 1, &f.fence);

        // Upload the game frame.
        const HostTexture *tex = nullptr;
        if (g_draw.valid)
            if (auto it = g_textures.find(g_draw.texture); it != g_textures.end() && it->second.width > 0)
                tex = &it->second;
        bool haveImage = false;
        if (tex && ensureFrameImage(f, tex->width, tex->height))
        {
            std::memcpy(f.stagingPtr, tex->pixels.data(), std::min<size_t>(tex->pixels.size(), static_cast<size_t>(f.stagingSize)));
            haveImage = true;
        }

        VkCommandBuffer cmd = f.cmd;
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);

        const VkImage swapImage = g_vk.swapImages[imageIndex];
        if (haveImage)
        {
            imageBarrier(cmd, f.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {static_cast<uint32_t>(f.imageWidth), static_cast<uint32_t>(f.imageHeight), 1u};
            vkCmdCopyBufferToImage(cmd, f.staging, f.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            imageBarrier(cmd, f.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        }

        imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkClearColorValue clear{};
        clear.float32[0] = g_clearColor.r / 255.0f;
        clear.float32[1] = g_clearColor.g / 255.0f;
        clear.float32[2] = g_clearColor.b / 255.0f;
        clear.float32[3] = 1.0f;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);

        if (haveImage)
        {
            // The runtime computed the destination rectangle from GetScreenWidth/Height, which report
            // the swapchain size, so it maps directly. Clamp to be safe.
            auto clampi = [](float v, int lo, int hi) { return std::clamp(static_cast<int>(v + 0.5f), lo, hi); };
            const int ew = static_cast<int>(g_vk.swapExtent.width), eh = static_cast<int>(g_vk.swapExtent.height);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[0] = {clampi(g_draw.src.x, 0, f.imageWidth), clampi(g_draw.src.y, 0, f.imageHeight), 0};
            blit.srcOffsets[1] = {clampi(g_draw.src.x + g_draw.src.width, 0, f.imageWidth), clampi(g_draw.src.y + g_draw.src.height, 0, f.imageHeight), 1};
            blit.dstOffsets[0] = {clampi(g_draw.dst.x, 0, ew), clampi(g_draw.dst.y, 0, eh), 0};
            blit.dstOffsets[1] = {clampi(g_draw.dst.x + g_draw.dst.width, 0, ew), clampi(g_draw.dst.y + g_draw.dst.height, 0, eh), 1};
            if (blit.srcOffsets[1].x > blit.srcOffsets[0].x && blit.srcOffsets[1].y > blit.srcOffsets[0].y &&
                blit.dstOffsets[1].x > blit.dstOffsets[0].x && blit.dstOffsets[1].y > blit.dstOffsets[0].y)
                vkCmdBlitImage(cmd, f.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &blit, g_vk.linearFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
        }

        imageBarrier(cmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        vkEndCommandBuffer(cmd);

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &f.imageAvailable;
        si.pWaitDstStageMask = &waitStage;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &g_vk.renderDone[imageIndex];
        r = vkQueueSubmit(g_vk.queue, 1, &si, f.fence);
        if (r != VK_SUCCESS)
        {
            logVk("vkQueueSubmit", r);
            return;
        }

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &g_vk.renderDone[imageIndex];
        pi.swapchainCount = 1;
        pi.pSwapchains = &g_vk.swapchain;
        pi.pImageIndices = &imageIndex;
        r = vkQueuePresentKHR(g_vk.queue, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
            g_vk.swapchainDirty = true;
        else if (r != VK_SUCCESS)
            logVk("vkQueuePresentKHR", r);
        g_vk.frameIndex = (g_vk.frameIndex + 1u) % kFramesInFlight;
    }

    // ---------------------------------------------------------------- input
    int toRaylibButton(SDL_GamepadButton b)
    {
        switch (b)
        {
        case SDL_GAMEPAD_BUTTON_DPAD_UP: return GAMEPAD_BUTTON_LEFT_FACE_UP;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return GAMEPAD_BUTTON_LEFT_FACE_RIGHT;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return GAMEPAD_BUTTON_LEFT_FACE_DOWN;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return GAMEPAD_BUTTON_LEFT_FACE_LEFT;
        case SDL_GAMEPAD_BUTTON_NORTH: return GAMEPAD_BUTTON_RIGHT_FACE_UP;
        case SDL_GAMEPAD_BUTTON_EAST: return GAMEPAD_BUTTON_RIGHT_FACE_RIGHT;
        case SDL_GAMEPAD_BUTTON_SOUTH: return GAMEPAD_BUTTON_RIGHT_FACE_DOWN;
        case SDL_GAMEPAD_BUTTON_WEST: return GAMEPAD_BUTTON_RIGHT_FACE_LEFT;
        case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return GAMEPAD_BUTTON_LEFT_TRIGGER_1;
        case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return GAMEPAD_BUTTON_RIGHT_TRIGGER_1;
        case SDL_GAMEPAD_BUTTON_BACK: return GAMEPAD_BUTTON_MIDDLE_LEFT;
        case SDL_GAMEPAD_BUTTON_GUIDE: return GAMEPAD_BUTTON_MIDDLE;
        case SDL_GAMEPAD_BUTTON_START: return GAMEPAD_BUTTON_MIDDLE_RIGHT;
        case SDL_GAMEPAD_BUTTON_LEFT_STICK: return GAMEPAD_BUTTON_LEFT_THUMB;
        case SDL_GAMEPAD_BUTTON_RIGHT_STICK: return GAMEPAD_BUTTON_RIGHT_THUMB;
        default: return GAMEPAD_BUTTON_UNKNOWN;
        }
    }

    void openFirstGamepad()
    {
        if (g_gamepad)
            return;
        int count = 0;
        SDL_JoystickID *ids = SDL_GetGamepads(&count);
        if (ids && count > 0)
        {
            g_gamepad = SDL_OpenGamepad(ids[0]);
            if (g_gamepad)
                std::fprintf(stderr, "[host] gamepad: %s\n", SDL_GetGamepadName(g_gamepad));
        }
        SDL_free(ids);
    }

    void sampleInput()
    {
        int numKeys = 0;
        const bool *state = SDL_GetKeyboardState(&numKeys);
        for (int k = 0; k < kMaxKeys; ++k)
        {
            const uint8_t down = (state && k < numKeys && state[k]) ? 1u : 0u;
            g_keyPressed[k] = (down && !g_keyPrev[k]) ? 1u : 0u;
            g_keyPrev[k] = down;
            g_keyDown[k].store(down, std::memory_order_relaxed);
        }

        if (!g_gamepad)
            openFirstGamepad();
        if (g_gamepad && !SDL_GamepadConnected(g_gamepad))
        {
            SDL_CloseGamepad(g_gamepad);
            g_gamepad = nullptr;
            openFirstGamepad();
        }
        g_gamepadPresent.store(g_gamepad != nullptr, std::memory_order_relaxed);
        uint32_t buttons = 0;
        if (g_gamepad)
        {
            for (int b = 0; b < SDL_GAMEPAD_BUTTON_COUNT; ++b)
                if (SDL_GetGamepadButton(g_gamepad, static_cast<SDL_GamepadButton>(b)))
                {
                    const int rb = toRaylibButton(static_cast<SDL_GamepadButton>(b));
                    if (rb != GAMEPAD_BUTTON_UNKNOWN)
                        buttons |= 1u << rb;
                }
            const Sint16 lt = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER);
            const Sint16 rt = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER);
            if (lt > 8000)
                buttons |= 1u << GAMEPAD_BUTTON_LEFT_TRIGGER_2;
            if (rt > 8000)
                buttons |= 1u << GAMEPAD_BUTTON_RIGHT_TRIGGER_2;
            g_padAxes[GAMEPAD_AXIS_LEFT_X].store(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFTX), std::memory_order_relaxed);
            g_padAxes[GAMEPAD_AXIS_LEFT_Y].store(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFTY), std::memory_order_relaxed);
            g_padAxes[GAMEPAD_AXIS_RIGHT_X].store(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHTX), std::memory_order_relaxed);
            g_padAxes[GAMEPAD_AXIS_RIGHT_Y].store(SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHTY), std::memory_order_relaxed);
            g_padAxes[GAMEPAD_AXIS_LEFT_TRIGGER].store(lt, std::memory_order_relaxed);
            g_padAxes[GAMEPAD_AXIS_RIGHT_TRIGGER].store(rt, std::memory_order_relaxed);
        }
        else
        {
            for (auto &a : g_padAxes)
                a.store(0, std::memory_order_relaxed);
        }
        g_padButtons.store(buttons, std::memory_order_relaxed);
    }

    void pumpEvents()
    {
        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            switch (e.type)
            {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                g_shouldClose = true;
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_RESIZED:
            case SDL_EVENT_WINDOW_RESTORED:
                g_vk.swapchainDirty = true;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (e.key.scancode == SDL_SCANCODE_RETURN && (e.key.mod & SDL_KMOD_ALT) && !e.key.repeat)
                {
                    const bool fullscreen = (SDL_GetWindowFlags(g_window) & SDL_WINDOW_FULLSCREEN) != 0;
                    SDL_SetWindowFullscreen(g_window, !fullscreen);
                    g_vk.swapchainDirty = true;
                }
                break;
            case SDL_EVENT_GAMEPAD_ADDED:
                openFirstGamepad();
                break;
            default:
                break;
            }
        }
        sampleInput();
    }

    // ---------------------------------------------------------------- audio
    SDL_AudioDeviceID g_audioDevice = 0;
    int g_audioBufferFrames = 1024;

    struct HostStream
    {
        SDL_AudioStream *stream = nullptr;
        AudioCallback callback = nullptr;
        unsigned int channels = 2;
        unsigned int bytesPerSample = 2;
        std::vector<uint8_t> scratch;
    };

    void SDLCALL streamGetCallback(void *userdata, SDL_AudioStream *stream, int additionalAmount, int /*totalAmount*/)
    {
        HostStream *hs = static_cast<HostStream *>(userdata);
        if (!hs || !hs->callback || additionalAmount <= 0)
            return;
        const unsigned int frameBytes = hs->channels * hs->bytesPerSample;
        // Produce at least a small block so the callback is not asked for a few bytes at a time.
        const unsigned int frames = std::max<unsigned int>((static_cast<unsigned int>(additionalAmount) + frameBytes - 1u) / frameBytes, 256u);
        hs->scratch.assign(static_cast<size_t>(frames) * frameBytes, 0u);
        hs->callback(hs->scratch.data(), frames);
        SDL_PutAudioStreamData(stream, hs->scratch.data(), static_cast<int>(hs->scratch.size()));
    }

    SDL_AudioFormat formatForSampleSize(unsigned int bits)
    {
        return bits == 8 ? SDL_AUDIO_U8 : bits == 32 ? SDL_AUDIO_F32 : SDL_AUDIO_S16;
    }

    struct HostWave
    {
        SDL_AudioSpec spec{};
        Uint8 *data = nullptr;
        Uint32 bytes = 0;
    };
    struct HostSound
    {
        SDL_AudioStream *stream = nullptr;
        std::vector<uint8_t> data;
    };
}

// ================================================================ window / frame
void SetConfigFlags(unsigned int flags) { g_configFlags |= flags; }

void InitWindow(int width, int height, const char *title)
{
    SDL_SetMainReady();
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD))
    {
        std::fprintf(stderr, "[host] SDL_Init failed: %s\n", SDL_GetError());
        return;
    }
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN;
    if (g_configFlags & FLAG_WINDOW_RESIZABLE)
        flags |= SDL_WINDOW_RESIZABLE;
    if (g_configFlags & FLAG_FULLSCREEN_MODE)
        flags |= SDL_WINDOW_FULLSCREEN;
    g_window = SDL_CreateWindow(title ? title : "PS2Recomp", width, height, flags);
    bool vulkanWindow = g_window != nullptr;
    if (!g_window)
    {
        std::fprintf(stderr, "[host] SDL_CreateWindow (Vulkan) failed: %s\n", SDL_GetError());
        g_window = SDL_CreateWindow(title ? title : "PS2Recomp", width, height, flags & ~SDL_WINDOW_VULKAN);
        if (!g_window)
        {
            std::fprintf(stderr, "[host] SDL_CreateWindow failed: %s\n", SDL_GetError());
            return;
        }
    }
    if (!vulkanWindow || !initVulkan())
    {
        std::fprintf(stderr, "[host] Vulkan initialisation failed; the window will stay black\n");
    }
    g_windowReady = true;
    g_nextFrameNs = SDL_GetTicksNS();
    pumpEvents();
}

void CloseWindow()
{
    if (!g_window)
        return;
    shutdownVulkan();
    if (g_gamepad)
        SDL_CloseGamepad(g_gamepad);
    g_gamepad = nullptr;
    SDL_DestroyWindow(g_window);
    g_window = nullptr;
    g_windowReady = false;
    SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD);
}

bool WindowShouldClose() { return g_shouldClose; }
bool IsWindowReady() { return g_windowReady; }
void SetTargetFPS(int fps) { g_targetFps = fps; }
void SetTraceLogLevel(int) {}

int GetScreenWidth()
{
    if (g_vk.swapchain)
        return static_cast<int>(g_vk.swapExtent.width);
    int w = 0, h = 0;
    if (g_window)
        SDL_GetWindowSizeInPixels(g_window, &w, &h);
    return w;
}

int GetScreenHeight()
{
    if (g_vk.swapchain)
        return static_cast<int>(g_vk.swapExtent.height);
    int w = 0, h = 0;
    if (g_window)
        SDL_GetWindowSizeInPixels(g_window, &w, &h);
    return h;
}

void BeginDrawing()
{
    g_draw.valid = false;
    // A resize may be pending: recreate now so the runtime lays the frame out for the new size.
    if (g_vk.ok && g_vk.swapchainDirty)
        createSwapchain();
}

void ClearBackground(Color color) { g_clearColor = color; }

void EndDrawing()
{
    presentFrame();
    pumpEvents();
    if (g_targetFps > 0)
    {
        const uint64_t period = 1000000000ull / static_cast<uint64_t>(g_targetFps);
        g_nextFrameNs += period;
        const uint64_t now = SDL_GetTicksNS();
        if (g_nextFrameNs > now)
            SDL_DelayPrecise(g_nextFrameNs - now);
        else if (now - g_nextFrameNs > period * 4u)
            g_nextFrameNs = now; // fell far behind: do not try to catch up
    }
}

Image GenImageColor(int width, int height, Color color)
{
    Image img{};
    img.width = width;
    img.height = height;
    img.mipmaps = 1;
    img.format = 7; // raylib PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    const size_t count = static_cast<size_t>(std::max(width, 0)) * static_cast<size_t>(std::max(height, 0));
    uint8_t *data = static_cast<uint8_t *>(std::malloc(count * 4u));
    for (size_t i = 0; data && i < count; ++i)
    {
        data[i * 4 + 0] = color.r;
        data[i * 4 + 1] = color.g;
        data[i * 4 + 2] = color.b;
        data[i * 4 + 3] = color.a;
    }
    img.data = data;
    return img;
}

void UnloadImage(Image image) { std::free(image.data); }

Texture2D LoadTextureFromImage(Image image)
{
    Texture2D t{};
    t.id = g_nextTextureId++;
    t.width = image.width;
    t.height = image.height;
    t.mipmaps = 1;
    t.format = 7;
    HostTexture &ht = g_textures[t.id];
    ht.width = image.width;
    ht.height = image.height;
    ht.pixels.assign(static_cast<size_t>(image.width) * image.height * 4u, 0u);
    if (image.data)
        std::memcpy(ht.pixels.data(), image.data, ht.pixels.size());
    return t;
}

void UpdateTexture(Texture2D texture, const void *pixels)
{
    auto it = g_textures.find(texture.id);
    if (it == g_textures.end() || !pixels)
        return;
    std::memcpy(it->second.pixels.data(), pixels, it->second.pixels.size());
    it->second.dirty = true;
}

void UnloadTexture(Texture2D texture) { g_textures.erase(texture.id); }

void DrawTexturePro(Texture2D texture, Rectangle source, Rectangle dest, Vector2, float, Color)
{
    g_draw.texture = texture.id;
    g_draw.src = source;
    g_draw.dst = dest;
    g_draw.valid = true;
}

// ================================================================ input
bool IsKeyDown(int key)
{
    return key > 0 && key < kMaxKeys && g_keyDown[key].load(std::memory_order_relaxed) != 0u;
}

bool IsKeyPressed(int key)
{
    return key > 0 && key < kMaxKeys && g_keyPressed[key] != 0u;
}

bool IsGamepadAvailable(int gamepad)
{
    return gamepad == 0 && g_gamepadPresent.load(std::memory_order_relaxed);
}

bool IsGamepadButtonDown(int gamepad, int button)
{
    if (gamepad != 0 || button <= 0 || button >= 32)
        return false;
    return (g_padButtons.load(std::memory_order_relaxed) & (1u << button)) != 0u;
}

float GetGamepadAxisMovement(int gamepad, int axis)
{
    if (gamepad != 0 || axis < 0 || axis >= 6)
        return 0.0f;
    const int v = g_padAxes[axis].load(std::memory_order_relaxed);
    if (axis >= GAMEPAD_AXIS_LEFT_TRIGGER)
        return (v / 32767.0f) * 2.0f - 1.0f; // raylib reports triggers as -1 (released) .. 1
    return std::max(-1.0f, v / 32767.0f);
}

// ================================================================ audio
void InitAudioDevice()
{
    if (g_audioDevice)
        return;
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO))
    {
        std::fprintf(stderr, "[host] SDL audio init failed: %s\n", SDL_GetError());
        return;
    }
    char frames[16];
    std::snprintf(frames, sizeof(frames), "%d", g_audioBufferFrames);
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, frames);
    g_audioDevice = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, nullptr);
    if (!g_audioDevice)
    {
        std::fprintf(stderr, "[host] no audio output device: %s\n", SDL_GetError());
        return;
    }
    SDL_AudioSpec spec{};
    int sampleFrames = 0;
    SDL_GetAudioDeviceFormat(g_audioDevice, &spec, &sampleFrames);
    std::fprintf(stderr, "[host] audio device: %s, %d Hz, %d ch, %d-frame buffer\n",
                 SDL_GetAudioDeviceName(g_audioDevice), spec.freq, spec.channels, sampleFrames);
    SDL_ResumeAudioDevice(g_audioDevice);
}

void CloseAudioDevice()
{
    if (g_audioDevice)
        SDL_CloseAudioDevice(g_audioDevice);
    g_audioDevice = 0;
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

bool IsAudioDeviceReady() { return g_audioDevice != 0; }

void SetAudioStreamBufferSizeDefault(int size)
{
    if (size > 0)
        g_audioBufferFrames = size;
}

AudioStream LoadAudioStream(unsigned int sampleRate, unsigned int sampleSize, unsigned int channels)
{
    AudioStream as{};
    as.sampleRate = sampleRate;
    as.sampleSize = sampleSize;
    as.channels = channels;
    if (!g_audioDevice)
        return as;
    SDL_AudioSpec src{formatForSampleSize(sampleSize), static_cast<int>(channels), static_cast<int>(sampleRate)};
    HostStream *hs = new HostStream();
    hs->channels = channels;
    hs->bytesPerSample = sampleSize / 8u;
    hs->stream = SDL_CreateAudioStream(&src, nullptr);
    if (!hs->stream || !SDL_BindAudioStream(g_audioDevice, hs->stream))
    {
        std::fprintf(stderr, "[host] audio stream failed: %s\n", SDL_GetError());
        if (hs->stream)
            SDL_DestroyAudioStream(hs->stream);
        delete hs;
        return as;
    }
    as.buffer = hs;
    return as;
}

void UnloadAudioStream(AudioStream stream)
{
    HostStream *hs = static_cast<HostStream *>(stream.buffer);
    if (!hs)
        return;
    SDL_DestroyAudioStream(hs->stream);
    delete hs;
}

void SetAudioStreamCallback(AudioStream stream, AudioCallback callback)
{
    HostStream *hs = static_cast<HostStream *>(stream.buffer);
    if (!hs)
        return;
    SDL_LockAudioStream(hs->stream);
    hs->callback = callback;
    SDL_UnlockAudioStream(hs->stream);
    SDL_SetAudioStreamGetCallback(hs->stream, streamGetCallback, hs);
}

void PlayAudioStream(AudioStream)
{
    if (g_audioDevice)
        SDL_ResumeAudioDevice(g_audioDevice);
}

Wave LoadWaveFromMemory(const char *, const unsigned char *fileData, int dataSize)
{
    Wave w{};
    if (!fileData || dataSize <= 0)
        return w;
    HostWave *hw = new HostWave();
    SDL_IOStream *io = SDL_IOFromConstMem(fileData, static_cast<size_t>(dataSize));
    if (!io || !SDL_LoadWAV_IO(io, true, &hw->spec, &hw->data, &hw->bytes))
    {
        delete hw;
        return w;
    }
    const unsigned int frameBytes = static_cast<unsigned int>(SDL_AUDIO_FRAMESIZE(hw->spec));
    w.frameCount = frameBytes ? hw->bytes / frameBytes : 0u;
    w.sampleRate = static_cast<unsigned int>(hw->spec.freq);
    w.sampleSize = SDL_AUDIO_BITSIZE(hw->spec.format);
    w.channels = static_cast<unsigned int>(hw->spec.channels);
    w.data = hw;
    return w;
}

void UnloadWave(Wave wave)
{
    HostWave *hw = static_cast<HostWave *>(wave.data);
    if (!hw)
        return;
    SDL_free(hw->data);
    delete hw;
}

Sound LoadSoundFromWave(Wave wave)
{
    Sound s{};
    HostWave *hw = static_cast<HostWave *>(wave.data);
    if (!hw || !g_audioDevice)
        return s;
    HostSound *hs = new HostSound();
    hs->data.assign(hw->data, hw->data + hw->bytes);
    hs->stream = SDL_CreateAudioStream(&hw->spec, nullptr);
    if (!hs->stream || !SDL_BindAudioStream(g_audioDevice, hs->stream))
    {
        if (hs->stream)
            SDL_DestroyAudioStream(hs->stream);
        delete hs;
        return s;
    }
    s.stream.buffer = hs;
    s.stream.sampleRate = wave.sampleRate;
    s.stream.sampleSize = wave.sampleSize;
    s.stream.channels = wave.channels;
    s.frameCount = wave.frameCount;
    return s;
}

void UnloadSound(Sound sound)
{
    HostSound *hs = static_cast<HostSound *>(sound.stream.buffer);
    if (!hs)
        return;
    SDL_DestroyAudioStream(hs->stream);
    delete hs;
}

void PlaySound(Sound sound)
{
    HostSound *hs = static_cast<HostSound *>(sound.stream.buffer);
    if (!hs)
        return;
    SDL_ClearAudioStream(hs->stream);
    SDL_PutAudioStreamData(hs->stream, hs->data.data(), static_cast<int>(hs->data.size()));
    SDL_FlushAudioStream(hs->stream);
}

void StopSound(Sound sound)
{
    HostSound *hs = static_cast<HostSound *>(sound.stream.buffer);
    if (hs)
        SDL_ClearAudioStream(hs->stream);
}

bool IsSoundPlaying(Sound sound)
{
    HostSound *hs = static_cast<HostSound *>(sound.stream.buffer);
    return hs && SDL_GetAudioStreamQueued(hs->stream) > 0;
}

void SetSoundPitch(Sound sound, float pitch)
{
    HostSound *hs = static_cast<HostSound *>(sound.stream.buffer);
    if (hs && pitch > 0.0f)
        SDL_SetAudioStreamFrequencyRatio(hs->stream, std::clamp(pitch, 0.01f, 100.0f));
}

void SetSoundVolume(Sound sound, float volume)
{
    HostSound *hs = static_cast<HostSound *>(sound.stream.buffer);
    if (hs)
        SDL_SetAudioStreamGain(hs->stream, std::max(0.0f, volume));
}
