#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct GSPixelPipe;
struct GSTexSampler;

// Light lock for the backend's producer-side state: taken once per primitive on the EE thread and
// only occasionally by the host presentation thread, so a spin/yield lock beats std::mutex.
class GsLock
{
public:
    void lock() noexcept
    {
        for (uint32_t spins = 0; m_flag.exchange(true, std::memory_order_acquire);)
            while (m_flag.load(std::memory_order_relaxed))
            {
                if (++spins < 64u)
                    continue;
                std::this_thread::yield();
            }
    }
    bool try_lock() noexcept { return !m_flag.exchange(true, std::memory_order_acquire); }
    void unlock() noexcept { m_flag.store(false, std::memory_order_release); }

private:
    std::atomic<bool> m_flag{false};
};

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();
    ~GSCpuBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;
    void QueuePresentSnapshot(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    // ---- Threaded rasteriser -------------------------------------------------------------
    // Submit() only queues primitives; worker threads rasterise them, each owning an
    // interleaved set of 8-row screen bands. Every other operation that touches VRAM is queued
    // as an ordered "global" command (all workers meet at a barrier, worker 0 runs it), and
    // every read-back waits for the queue first (SyncUnlocked). PS2_GS_THREADS=0 disables it.
    struct Command
    {
        bool global = false;
        GSPrimitiveBatch batch{};
        std::shared_ptr<const std::array<uint32_t, 256>> palette; // decoded CLUT for indexed textures
        std::function<void()> fn;
        std::vector<GSVertex> more; // further primitives with the same state (3 vertices each)
    };
    struct alignas(64) WorkerSlot
    {
        std::atomic<uint64_t> done{0};
    };
    struct DirtyRange
    {
        uint64_t key;
        uint32_t start;
        uint32_t end;
    };
    static constexpr uint32_t kRingSize = 32768u;
    void StartWorkersUnlocked();
    void StopWorkers();
    void WorkerMain(uint32_t index);
    void EnqueueUnlocked(Command &&command);    // publishes the pending draw batch first
    void EnqueueRawUnlocked(Command &&command);
    void FlushPendingUnlocked() const;         // publish the draw batch being collected
    // Consecutive draws with identical state and palette are collected into one command (up to
    // kMaxBatchPrims) before being published, so the ring and every worker handle far fewer
    // commands. Mutable: syncs (const) publish it before waiting.
    static constexpr uint32_t kMaxBatchPrims = 64u;
    mutable Command m_pending;
    mutable bool m_hasPending = false;
    mutable uint32_t m_pendingPrims = 0;
    // `touchStart/End`: VRAM bytes the command itself writes (for CanRunDirectUnlocked).
    void EnqueueGlobalUnlocked(std::function<void()> fn, uint32_t touchStart = 0u, uint32_t touchEnd = 0x400000u);
    void SyncUnlocked(int reason) const;
    uint64_t HazardTargetUnlocked(uint32_t start, uint32_t end, bool readOnly) const;
    void SyncToUnlocked(uint64_t target, int reason) const;
    void NoteDrawHazardsUnlocked(const GSPrimitiveBatch &batch);
    uint64_t MinDone() const;
    bool threaded() const { return m_threadCount != 0u; }

    uint32_t m_threadCount = 0;      // 0 = synchronous
    int m_threadMode = -1;           // -1 = not decided yet
    std::unique_ptr<Command[]> m_ring;
    std::unique_ptr<WorkerSlot[]> m_workerDone;
    std::vector<std::thread> m_workers;
    alignas(64) std::atomic<uint64_t> m_writeIdx{0};
    std::atomic<bool> m_stopWorkers{false};
    std::atomic<uint32_t> m_sleepers{0};
    mutable std::mutex m_wakeMutex;
    mutable std::condition_variable m_wakeCv;
    mutable uint64_t m_lastWakeIdx = 0; // producer side: m_writeIdx at the last wake-up
    void WakeWorkers() const;
    std::vector<DirtyRange> m_dirty;  // target ranges written by draws queued since the last barrier
    std::vector<DirtyRange> m_reads;  // texture ranges read by draws queued since the last barrier
    struct Epoch
    {
        uint64_t globalIdx; // the barrier closing this epoch; done once every worker passed it
        std::vector<DirtyRange> ranges; // writes + reads of its draws, plus the barrier's own writes
    };
    std::vector<Epoch> m_epochs;  // closed epochs that may still be running
    // Asynchronous presentation (threaded mode): a queued barrier copies VRAM into m_presentStage
    // at its place in the command stream; Present() converts the latest completed copy, so the
    // host thread never waits for (or blocks) the producer.
    std::mutex m_presentMutex;
    std::vector<uint8_t> m_presentStage, m_presentLatest;
    GSPresentationRequest m_presentStageRequest{}, m_presentLatestRequest{};
    bool m_presentLatestNew = false;
    bool m_presentHaveAny = false;
    std::atomic<bool> m_presentPending{false};
    std::atomic<uint64_t> m_lastFlipSnapshotNs{0}; // when the last flip-anchored snapshot was queued
    void EnqueuePresentSnapshotUnlocked(const GSPresentationRequest &request);
    std::shared_ptr<const std::array<uint32_t, 256>> m_sharedPalette;
    uint64_t m_sharedPaletteVersion = 0;
    uint64_t m_sharedPaletteKey = ~0ull;
    // readOnly: the caller only reads [start, end), so queued readers of it do not conflict.
    bool CanRunDirectUnlocked(uint32_t start, uint32_t end, bool readOnly = false) const;
    uint64_t PaletteKey(const GSDrawState &state) const;

    struct PaletteCache
    {
        std::array<uint32_t, 256> palette{};
        uint64_t version = 0;
        uint64_t key = ~0ull;
    };
    std::array<PaletteCache, 17> m_paletteCaches{};

    void BeginTransferUnlocked(const GSTransferCommand &command);
    void UploadImageUnlocked(const uint8_t *data, uint32_t sizeBytes);
    void UploadImageImpl(const GSTransferCommand &xfer, GSTransferSnapshot &st,
                         const uint8_t *data, uint32_t sizeBytes, bool write);
    void ClearFramebufferUnlocked(const GSContext &context, uint32_t rgba);
    void ResetUnlocked();
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch);
    void DrawTriangle(const GSPrimitiveBatch &batch);
    void DrawLine(const GSPrimitiveBatch &batch);
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t LookupCLUT(const GSDrawState &state, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

    // Fast paths: per-primitive decoded state, used by triangles and sprites.
    void SetupPixelPipe(const GSDrawState &state, GSPixelPipe &pipe) const;
    void SetupSampler(const GSDrawState &state, GSTexSampler &sampler);
    void WritePixelFast(const GSPixelPipe &pipe, int x, int y, uint32_t z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleFast(const GSTexSampler &sampler, float s, float t, float q, uint16_t u, uint16_t v) const;
    uint32_t FetchTexel(const GSTexSampler &sampler, int u, int v) const;

    void PerformLocalToLocalTransfer();
    void CopyLocalToLocal(const GSTransferCommand &transfer);
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable GsLock m_mutex; // producer-side state; nearly uncontended (EE thread, host present)
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    std::array<uint16_t, 512> m_clut{};
    std::array<uint32_t, 2> m_clutCbp{};
    uint64_t m_clutVersion = 1;
    GSMem::TexturePageCache m_texturePageCache;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};
