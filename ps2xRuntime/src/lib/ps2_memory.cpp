#include "runtime/ps2_memory.h"
#include "runtime/ps2_address.h"
#include "runtime/gs/gs_frontend.h"
#include "ps2_log.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <deque>
#include <cstdlib>
#include <thread>
#include "ThreadNaming.h"
#if defined(_WIN32)
extern "C" __declspec(dllimport) int __stdcall SetThreadPriority(void *hThread, int nPriority);
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <immintrin.h>
#endif

void ssx3FrameRecord(PS2Memory &mem, uint32_t type, const void *hdr, uint32_t hdrSize, const uint8_t *data, uint32_t size); // ps2_vif1_interpreter.cpp

namespace
{
    inline void inRange(uint32_t offset, size_t bytes, size_t regionSize, const char *op, uint32_t address)
    {
        if (static_cast<uint64_t>(offset) + static_cast<uint64_t>(bytes) > static_cast<uint64_t>(regionSize))
        {
            throw std::runtime_error(std::string(op) + " out-of-bounds at address: 0x" + std::to_string(address));
        }
    }

    template <typename T>
    inline T loadScalar(const uint8_t *base, uint32_t offset, size_t regionSize, const char *op, uint32_t address)
    {
        inRange(offset, sizeof(T), regionSize, op, address);
        T value{};
        std::memcpy(&value, base + offset, sizeof(T));
        return value;
    }

    template <typename T>
    inline void storeScalar(uint8_t *base, uint32_t offset, size_t regionSize, T value, const char *op, uint32_t address)
    {
        inRange(offset, sizeof(T), regionSize, op, address);
        std::memcpy(base + offset, &value, sizeof(T));
    }

}

// ---- Asynchronous VIF1/GIF worker ---------------------------------------------------------------
// VIF1 and GIF (PATH3) DMA kicks from the EE are completed immediately (as before: the DMA data is
// captured into owned buffers at kick time) and their processing - VIF1 unpacks, VU1 micro
// programs, GS front end - runs on one worker thread in kick order. EE accesses that could observe
// that state (VU1 memory, VIF1 registers, GS privileged registers other than CSR status, GS
// library calls, synchronous transfers) wait for the worker first (gpuSync). PS2_ASYNC_GPU=0
// processes everything on the EE thread as before. Small GS read-backs (lens flare depth probes)
// do not wait: see asyncReadbackMode.
struct PS2Memory::GpuJob
{
    std::vector<std::vector<uint8_t>> gif;  // PATH3 packets
    std::vector<std::vector<uint8_t>> vif1; // VIF1 streams
    std::function<void()> fn;               // extra work run after the streams
    bool drain = true;
    bool countKick = false;
    uint64_t enqueueNs = 0; // for the worker's scheduling-latency statistic
    uint32_t finishEpoch = 0; // see s_finishEpoch
};

void vif1Observe(uint32_t) {} // old overlap measurement hook (ps2_runtime.cpp still calls it)
void (*g_ps2GpuSyncHook)() = nullptr; // called by GS entry points used from the EE (gs_frontend.cpp)
extern void (*g_ps2GsFinishHook)(std::atomic<uint64_t> &csr); // gs_frontend.cpp: a FINISH register write
void (*g_ps2FlipHook)() = nullptr;    // set by PS2Runtime: GS::notePresentPoint (a DISPFB write = the game's flip)
// set by PS2Runtime: copies up to `bytes` of pending GS local->host (readback) data to dst, returns the count.
uint32_t (*g_ps2GsReadbackHook)(uint8_t *dst, uint32_t bytes) = nullptr;

namespace
{
    // Recycled DMA capture buffers: VIF1 chains are hundreds of KB, and allocating and freeing
    // them per kick went through VirtualAlloc/VirtualFree (plus fresh-page faults) every frame.
    std::mutex s_bufPoolMutex;
    std::vector<std::vector<uint8_t>> s_bufPool;
    std::vector<uint8_t> gpuTakeBuffer()
    {
        std::lock_guard<std::mutex> lock(s_bufPoolMutex);
        if (s_bufPool.empty())
            return {};
        std::vector<uint8_t> v = std::move(s_bufPool.back());
        s_bufPool.pop_back();
        v.clear();
        return v;
    }
    void gpuReturnBuffers(std::vector<std::vector<uint8_t>> &buffers)
    {
        std::lock_guard<std::mutex> lock(s_bufPoolMutex);
        for (auto &b : buffers)
            if (b.capacity() >= 16384u && s_bufPool.size() < 16u)
                s_bufPool.push_back(std::move(b));
        buffers.clear();
    }

    std::atomic<uint64_t> s_vif1KickNs{0}, s_vif1Kicks{0};
    std::atomic<uint64_t> s_gpuSyncNs[PS2Memory::kGpuSyncReasonCount]{};
    std::atomic<uint64_t> s_gpuSyncCount[PS2Memory::kGpuSyncReasonCount]{};
    std::atomic<uint64_t> s_gpuBusyNs{0}, s_gpuBackpressureNs{0}, s_gpuJobs{0}, s_gpuLatencyNs{0};
    std::atomic<uint64_t> s_gsStageBusyNs{0}, s_gsStageWaitNs{0}, s_gsStageSyncNs{0}, s_gsStageSyncs{0};
    std::atomic<uint64_t> s_finishDropped{0}, s_finishEarly{0};
    std::atomic<uint64_t> s_rbServed{0}, s_rbMissed{0}, s_rbCovered{0}, s_rbDirect{0}, s_rbRefreshed{0}, s_rbAgeNs{0}, s_rbDist{0};

    inline uint64_t gpuNowNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    thread_local bool t_onGpuWorker = false;
    thread_local bool t_onGsStage = false;

    // CSR.FINISH without waiting. The EE clears FINISH (CSR write with bit 1) and then waits for
    // the FINISH of a packet it sends afterwards; a FINISH still in the queue from before the
    // clear must not count. The EE used to wait for the whole queue before every clear. Instead
    // each clear starts a new "epoch", queued work carries the epoch it was queued in, and a
    // FINISH register write from an older epoch is dropped - the same result as if the queue had
    // been drained before the clear, without the wait.
    std::mutex s_finishMutex;                // epoch change + CSR.FINISH updates
    std::atomic<uint32_t> s_finishEpoch{0};  // bumped under s_finishMutex
    thread_local uint32_t t_finishEpoch = 0; // epoch of the work this thread is processing
    thread_local bool t_finishEpochSet = false;
    bool gpuAsyncEnabled();
    // The GS front end's packet entry (set by PS2Runtime before any DMA runs).
    std::function<void(const uint8_t *, uint32_t)> g_gsStageProcess;

    // Second pipeline stage: the GIF packets the worker produces (VU1 XGKICKs, VIF1 DIRECT, PATH3)
    // go to a "GS thread" that runs the GS front end and renderer, so VU1 geometry and GS drawing
    // of a frame overlap instead of taking turns on the worker. Packets are copied into chunks
    // (many small XGKICK packets per chunk); closures (privileged register writes, flips) are
    // queued in order with them. Anything else that touches the GS waits for the queue to drain
    // (gsStageSync), as the EE already does for the worker. PS2_ASYNC_GS=0 keeps the GS on the
    // worker thread.
    struct GsStage
    {
        struct Item
        {
            std::vector<uint8_t> packets; // [u32 size][bytes padded to 4] ...
            std::function<void()> fn;
            uint32_t epoch = 0; // FINISH epoch of the EE job this came from
        };
        std::function<void(const uint8_t *, uint32_t)> process;
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<Item> queue;
        std::atomic<uint64_t> enqueued{0}, done{0};
        std::atomic<uint32_t> queued{0};
        std::vector<uint8_t> cur; // chunk being filled (worker thread only)
        uint32_t curEpoch = 0;
        std::thread thread;
        std::atomic<bool> started{false};

        static constexpr size_t kChunkBytes = 48u * 1024u;
        static constexpr uint64_t kMaxPending = 96u; // chunks/closures the worker may be ahead

        void run()
        {
            t_onGsStage = true;
            ThreadNaming::SetCurrentThreadName("GS thread");
#if defined(_WIN32)
            SetThreadPriority(GetCurrentThread(), 1 /* THREAD_PRIORITY_ABOVE_NORMAL */);
#endif
            for (;;)
            {
                if (queued.load(std::memory_order_acquire) == 0u)
                {
                    for (int i = 0; i < 4000 && queued.load(std::memory_order_acquire) == 0u; ++i)
                        _mm_pause();
                    const uint64_t spinUntil = gpuNowNs() + 500000u;
                    while (queued.load(std::memory_order_acquire) == 0u && gpuNowNs() < spinUntil)
                        std::this_thread::yield();
                }
                Item item;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&]()
                            { return !queue.empty(); });
                    item = std::move(queue.front());
                    queue.pop_front();
                    queued.fetch_sub(1u, std::memory_order_relaxed);
                }
                const uint64_t t0 = gpuNowNs();
                t_finishEpoch = item.epoch;
                t_finishEpochSet = true;
                const uint8_t *p = item.packets.data();
                const uint8_t *end = p + item.packets.size();
                while (p + 4 <= end)
                {
                    uint32_t size = 0;
                    std::memcpy(&size, p, 4);
                    p += 4;
                    if (size == 0u || p + size > end)
                        break;
                    process(p, size);
                    p += (size + 3u) & ~3u;
                }
                if (item.fn)
                    item.fn();
                if (item.packets.capacity() != 0u)
                {
                    std::vector<std::vector<uint8_t>> one;
                    one.push_back(std::move(item.packets));
                    gpuReturnBuffers(one);
                }
                item = Item{};
                s_gsStageBusyNs.fetch_add(gpuNowNs() - t0, std::memory_order_relaxed);
                done.fetch_add(1u, std::memory_order_release);
            }
        }

        void push(Item &&item)
        {
            // Keep the worker a bounded distance ahead of the GS thread.
            if (enqueued.load(std::memory_order_relaxed) - done.load(std::memory_order_acquire) >= kMaxPending)
            {
                const uint64_t t0 = gpuNowNs();
                while (enqueued.load(std::memory_order_relaxed) - done.load(std::memory_order_acquire) >= kMaxPending)
                    std::this_thread::yield();
                s_gsStageWaitNs.fetch_add(gpuNowNs() - t0, std::memory_order_relaxed);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                queue.push_back(std::move(item));
                enqueued.fetch_add(1u, std::memory_order_release);
                queued.fetch_add(1u, std::memory_order_release);
            }
            cv.notify_one();
        }

        // Worker thread: hand the chunk being filled to the GS thread.
        void flush()
        {
            if (cur.empty())
                return;
            Item item;
            item.packets = std::move(cur);
            item.epoch = curEpoch;
            cur = gpuTakeBuffer();
            push(std::move(item));
        }

        void addPacket(const uint8_t *data, uint32_t size)
        {
            if (curEpoch != t_finishEpoch)
            {
                flush(); // a chunk holds packets of one FINISH epoch
                curEpoch = t_finishEpoch;
            }
            const size_t at = cur.size();
            cur.resize(at + 4u + ((size + 3u) & ~3u));
            std::memcpy(cur.data() + at, &size, 4);
            std::memcpy(cur.data() + at + 4u, data, size);
            if (cur.size() >= kChunkBytes)
                flush();
        }

        bool idle() const
        {
            return done.load(std::memory_order_acquire) >= enqueued.load(std::memory_order_acquire);
        }

        void wait() const
        {
            for (int i = 0; i < 4000 && !idle(); ++i)
                _mm_pause();
            while (!idle())
                std::this_thread::yield();
        }
    };
    GsStage *g_gsStage = nullptr; // created with the worker, never destroyed

    bool gsStageEnabled()
    {
        static const bool enabled = []()
        {
            const char *v = std::getenv("PS2_ASYNC_GS");
            const bool on = gpuAsyncEnabled() && !(v && v[0] == '0');
            std::fprintf(stderr, "[gpu] GS on its own thread %s (PS2_ASYNC_GS=0 disables)\n", on ? "on" : "off");
            return on;
        }();
        return enabled;
    }

    bool gpuAsyncEnabled()
    {
        static const bool enabled = []()
        {
            const char *v = std::getenv("PS2_ASYNC_GPU");
            const bool on = !(v && v[0] == '0');
            std::fprintf(stderr, "[gpu] asynchronous VIF1/GIF worker %s (PS2_ASYNC_GPU=0 disables)\n", on ? "on" : "off");
            return on;
        }();
        return enabled;
    }

    struct GpuWorker
    {
        PS2Memory *memory = nullptr;
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<PS2Memory::GpuJob> queue;
        std::atomic<uint64_t> enqueued{0}, done{0};
        std::atomic<uint32_t> pendingVif{0};
        std::atomic<uint32_t> queued{0};
        std::thread thread;

        void run()
        {
            t_onGpuWorker = true;
            ThreadNaming::SetCurrentThreadName("GPU worker");
#if defined(_WIN32)
            // The worker is on the critical path; don't let the spinning raster threads starve it.
            SetThreadPriority(GetCurrentThread(), 1 /* THREAD_PRIORITY_ABOVE_NORMAL */);
#endif
            uint64_t lastEndNs = gpuNowNs();
            for (;;)
            {
                if (queued.load(std::memory_order_acquire) == 0u)
                {
                    // Stay awake briefly: the EE usually queues the next job within a few hundred us.
                    for (int i = 0; i < 4000 && queued.load(std::memory_order_acquire) == 0u; ++i)
                        _mm_pause();
                    const uint64_t spinUntil = gpuNowNs() + 500000u;
                    while (queued.load(std::memory_order_acquire) == 0u && gpuNowNs() < spinUntil)
                        std::this_thread::yield();
                }
                PS2Memory::GpuJob job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&]()
                            { return !queue.empty(); });
                    job = std::move(queue.front());
                    queue.pop_front();
                    queued.fetch_sub(1u, std::memory_order_relaxed);
                }
                const auto t0 = std::chrono::steady_clock::now();
                {
                    const uint64_t startNs = gpuNowNs();
                    const uint64_t readyNs = std::max(job.enqueueNs, lastEndNs);
                    if (startNs > readyNs)
                        s_gpuLatencyNs.fetch_add(startNs - readyNs, std::memory_order_relaxed);
                }
                const bool hadVif = job.countKick && !job.vif1.empty(); // matches gpuEnqueue's backpressure count
                t_finishEpoch = job.finishEpoch;
                t_finishEpochSet = true;
                memory->runGpuJob(job);
                gpuReturnBuffers(job.vif1);
                gpuReturnBuffers(job.gif);
                job = PS2Memory::GpuJob{};
                s_gpuBusyNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                                      std::memory_order_relaxed);
                s_gpuJobs.fetch_add(1u, std::memory_order_relaxed);
                if (hadVif)
                    pendingVif.fetch_sub(1u, std::memory_order_relaxed);
                lastEndNs = gpuNowNs();
                done.fetch_add(1u, std::memory_order_release);
            }
        }

        bool idle() const
        {
            return done.load(std::memory_order_acquire) >= enqueued.load(std::memory_order_acquire);
        }
    };
    GpuWorker *g_gpuWorker = nullptr; // created on first use, never destroyed (detached thread)

    void gpuSyncGlobal(uint32_t reason)
    {
        GpuWorker *w = g_gpuWorker;
        if (!w || t_onGsStage)
            return;
        GsStage *gs = g_gsStage;
        if (t_onGpuWorker)
        {
            // The worker itself needs the GS (a read-back, a library call): let the GS thread
            // catch up with what the worker has produced.
            if (gs && gs->started.load(std::memory_order_acquire))
            {
                gs->flush();
                if (!gs->idle())
                {
                    const uint64_t t0 = gpuNowNs();
                    gs->wait();
                    s_gsStageSyncNs.fetch_add(gpuNowNs() - t0, std::memory_order_relaxed);
                    s_gsStageSyncs.fetch_add(1u, std::memory_order_relaxed);
                }
            }
            return;
        }
        const uint64_t target = w->enqueued.load(std::memory_order_acquire);
        if (w->done.load(std::memory_order_acquire) >= target && (!gs || gs->idle()))
            return;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 4000 && w->done.load(std::memory_order_acquire) < target; ++i)
            _mm_pause();
        while (w->done.load(std::memory_order_acquire) < target)
            std::this_thread::yield();
        if (gs)
            gs->wait(); // the worker hands its last chunk over before a job counts as done
        s_gpuSyncCount[reason].fetch_add(1u, std::memory_order_relaxed);
        s_gpuSyncNs[reason].fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                                      std::memory_order_relaxed);
    }

    bool gpuBusy()
    {
        return g_gpuWorker && (!g_gpuWorker->idle() || (g_gsStage && !g_gsStage->idle()));
    }
}

// GS thread entry points for PS2Runtime (declared at their use: ps2_memory.h is included by all
// recompiled code and stays unchanged).
void ps2GsStageSetProcess(std::function<void(const uint8_t *, uint32_t)> process);
bool ps2GsStageSubmit(const uint8_t *data, uint32_t sizeBytes);

void ps2GsStageSetProcess(std::function<void(const uint8_t *, uint32_t)> process)
{
    g_gsStageProcess = std::move(process);
}

// A GIF packet produced on the GPU worker goes to the GS thread; false = the caller processes
// it itself (no GS thread, or a caller that has already waited for it).
bool ps2GsStageSubmit(const uint8_t *data, uint32_t sizeBytes)
{
    if (!g_gsStage || !t_onGpuWorker)
        return false;
    g_gsStage->addPacket(data, sizeBytes);
    return true;
}

void PS2Memory::gpuSync(uint32_t reason)
{
    gpuSyncGlobal(reason < kGpuSyncReasonCount ? reason : kGpuSyncOther);
}

namespace
{
    void gpuEnqueue(PS2Memory *memory, PS2Memory::GpuJob &&job)
    {
        if (!g_gpuWorker)
        {
            g_gpuWorker = new GpuWorker();
            g_gpuWorker->memory = memory;
            if (gsStageEnabled() && g_gsStageProcess)
            {
                g_gsStage = new GsStage();
                g_gsStage->process = g_gsStageProcess;
                g_gsStage->thread = std::thread([]()
                                                { g_gsStage->run(); });
                g_gsStage->thread.detach();
                g_gsStage->started.store(true, std::memory_order_release);
            }
            g_ps2GpuSyncHook = []()
            { gpuSyncGlobal(PS2Memory::kGpuSyncGsCall); };
            g_ps2GsFinishHook = [](std::atomic<uint64_t> &csr)
            {
                std::lock_guard<std::mutex> lock(s_finishMutex);
                if (!t_finishEpochSet || t_finishEpoch == s_finishEpoch.load(std::memory_order_relaxed))
                    csr.fetch_or(0x2u);
                else
                    s_finishDropped.fetch_add(1u, std::memory_order_relaxed);
            };
            g_gpuWorker->thread = std::thread([]()
                                              { g_gpuWorker->run(); });
            g_gpuWorker->thread.detach();
        }
        GpuWorker &w = *g_gpuWorker;
        // Backpressure counts VIF1 DMA kicks only (display lists), not small FIFO writes, so a
        // qword written to the VIF1 FIFO right after a big kick does not wait for that kick.
        const bool hasVif = job.countKick && !job.vif1.empty();
        if (hasVif)
        {
            // Keep the EE at most a couple of display lists ahead of the worker.
            constexpr uint32_t kMaxPendingVif = 3u;
            if (w.pendingVif.load(std::memory_order_acquire) >= kMaxPendingVif)
            {
                const auto t0 = std::chrono::steady_clock::now();
                while (w.pendingVif.load(std::memory_order_acquire) >= kMaxPendingVif)
                    std::this_thread::yield();
                s_gpuBackpressureNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                                              std::memory_order_relaxed);
            }
            w.pendingVif.fetch_add(1u, std::memory_order_relaxed);
        }
        job.enqueueNs = gpuNowNs();
        job.finishEpoch = s_finishEpoch.load(std::memory_order_acquire);
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            w.queue.push_back(std::move(job));
            w.enqueued.fetch_add(1u, std::memory_order_release);
            w.queued.fetch_add(1u, std::memory_order_release);
        }
        w.cv.notify_one();
    }
}

namespace
{
    // ---- GS read-backs without waiting ---------------------------------------------------------
    // SSX 3 tests each lens flare's visibility by reading a 16x8 block of the Z buffer back from
    // the GS, ten or more times per frame in some scenes:
    //   CSR = FINISH (clear) / a GIF packet with just FINISH / wait for CSR.FINISH / clear it /
    //   a GIF packet BITBLTBUF, TRXPOS, TRXREG, TRXDIR=1 / VIF1_STAT.FDR / BUSDIR=1 /
    //   VIF1 DMA towards memory / VIF1_STAT=0 / BUSDIR=0
    // Each of those made the EE wait until the worker and the GS thread had caught up with
    // everything queued, so the three threads ran one after the other in those frames. Now:
    //  - clearing FINISH does not wait (see s_finishEpoch);
    //  - a packet that is only a FINISH write reports FINISH as soon as it is queued. FINISH means
    //    "everything before this has been drawn", and whatever the EE does next that depends on
    //    that (read-backs, GS memory, privileged registers) waits for the queue by itself;
    //  - the VIF1_STAT write (only FDR is writable) is just stored;
    //  - the download is answered from the most recent finished read of the same size at (almost)
    //    the same position - one frame old in practice - and the read that was asked for runs in
    //    queue order on the GS thread and becomes the answer for the next frame. With no such
    //    earlier read (first frame of a light, a camera cut) a depth probe is answered as
    //    "covered" (nearest depth everywhere: a flare starts one frame late rather than flashing
    //    through terrain); other formats wait as before.
    // PS2_ASYNC_READBACK=0 restores the waits; =1 also waits when there is no earlier read.
    int asyncReadbackMode()
    {
        static const int mode = []()
        {
            const char *v = std::getenv("PS2_ASYNC_READBACK");
            const int m = !gpuAsyncEnabled() ? 0 : (v && v[0] == '0') ? 0 : (v && v[0] == '1') ? 1 : 2;
            std::fprintf(stderr, "[gpu] GS read-backs answered from the previous frame: %s (PS2_ASYNC_READBACK=0 off, 1 wait for a first read, 2 never wait)\n",
                         m == 0 ? "off" : m == 1 ? "on, a first read waits" : "on, no waits");
            return m;
        }();
        return mode;
    }
    inline bool asyncReadbackEnabled()
    {
        return asyncReadbackMode() != 0;
    }

    struct ReadbackRequest
    {
        bool valid = false;
        uint32_t sbp = 0, sbw = 0, psm = 0, w = 0, h = 0;
        int32_t x = 0, y = 0;
        uint32_t pixelBytes = 0;
    };
    ReadbackRequest s_rbRequest; // the download the EE has set up (EE thread only)

    struct ReadbackEntry
    {
        ReadbackRequest key;
        uint64_t seq = 0, ns = 0;
        bool ready = false;
        std::vector<uint8_t> data;
    };
    constexpr size_t kRbEntries = 64u;
    constexpr uint32_t kRbMaxPixels = 4096u;          // small probes only; big reads (screenshots) wait
    constexpr int32_t kRbMaxDistance = 32;            // pixels the block may have moved since
    // Results older than this are not used (PS2_ASYNC_READBACK_MAXAGE_MS, default 250).
    uint64_t rbMaxAgeNs()
    {
        static const uint64_t ns = []()
        {
            const char *v = std::getenv("PS2_ASYNC_READBACK_MAXAGE_MS");
            const long ms = v ? std::atol(v) : 250;
            return static_cast<uint64_t>(ms > 0 ? ms : 250) * 1000000ull;
        }();
        return ns;
    }
    std::mutex s_rbMutex;
    ReadbackEntry s_rb[kRbEntries];
    uint64_t s_rbSeq = 0;

    enum class RbPacket
    {
        None,
        Finish,
        Download
    };

    // Is this PATH3 DMA one of the two packets of the read-back sequence?
    RbPacket rbClassify(const uint8_t *p, uint32_t qwc, ReadbackRequest &req)
    {
        if (qwc != 2u && qwc != 5u)
            return RbPacket::None;
        uint64_t q[10] = {};
        std::memcpy(q, p, static_cast<size_t>(qwc) * 16u);
        // GIF tag: NLOOP = qwc - 1, PACKED, one register, A+D.
        if ((q[0] & 0x7FFFull) != qwc - 1u || ((q[0] >> 58) & 0x3ull) != 0ull || (q[0] >> 60) != 1ull || (q[1] & 0xFull) != 0xEull)
            return RbPacket::None;
        if (qwc == 2u)
            return (q[3] & 0xFFull) == 0x61ull ? RbPacket::Finish : RbPacket::None;
        if ((q[3] & 0xFFull) != 0x50ull || (q[5] & 0xFFull) != 0x51ull || (q[7] & 0xFFull) != 0x52ull || (q[9] & 0xFFull) != 0x53ull ||
            (q[8] & 0x3ull) != 1ull)
            return RbPacket::None;
        req.sbp = static_cast<uint32_t>(q[2] & 0x3FFFull);
        req.sbw = static_cast<uint32_t>((q[2] >> 16) & 0x3Full);
        req.psm = static_cast<uint32_t>((q[2] >> 24) & 0x3Full);
        req.x = static_cast<int32_t>(q[4] & 0x7FFull);
        req.y = static_cast<int32_t>((q[4] >> 16) & 0x7FFull);
        req.w = static_cast<uint32_t>(q[6] & 0xFFFull);
        req.h = static_cast<uint32_t>((q[6] >> 32) & 0xFFFull);
        uint32_t bpp = 0u;
        switch (req.psm)
        {
        case 0x00u: // CT32
        case 0x30u: // Z32
            bpp = 32u;
            break;
        case 0x01u: // CT24
        case 0x31u: // Z24
            bpp = 24u;
            break;
        case 0x02u: // CT16
        case 0x0Au: // CT16S
        case 0x32u: // Z16
        case 0x3Au: // Z16S
            bpp = 16u;
            break;
        default:
            return RbPacket::None;
        }
        const uint32_t pixels = req.w * req.h;
        if (pixels == 0u || pixels > kRbMaxPixels)
            return RbPacket::None;
        req.pixelBytes = (pixels * bpp + 7u) / 8u;
        req.valid = true;
        return RbPacket::Download;
    }

    inline bool rbSameKind(const ReadbackRequest &a, const ReadbackRequest &b)
    {
        return a.sbp == b.sbp && a.sbw == b.sbw && a.psm == b.psm && a.w == b.w && a.h == b.h;
    }

    // Slot for a new entry: the oldest one (s_rbMutex held).
    ReadbackEntry &rbNewEntry(const ReadbackRequest &req, uint64_t now)
    {
        ReadbackEntry *e = &s_rb[0];
        for (ReadbackEntry &c : s_rb)
            if (c.seq < e->seq)
                e = &c;
        e->key = req;
        e->seq = ++s_rbSeq;
        e->ns = now;
        e->ready = false;
        return *e;
    }

    // A read that was done directly (the EE waited, or nothing was queued): remember the result.
    void rbStore(const ReadbackRequest &req, const uint8_t *data, uint32_t bytes)
    {
        std::lock_guard<std::mutex> lock(s_rbMutex);
        ReadbackEntry &e = rbNewEntry(req, gpuNowNs());
        e.data.assign(data, data + bytes);
        e.ready = true;
    }

    // Answer the download `req` (`bytes` of DMA data) from an earlier read and queue the real
    // read as the next answer. False: there is nothing to answer with, the caller reads directly.
    bool rbServe(PS2Memory *memory, const ReadbackRequest &req, uint8_t *dst, uint32_t bytes)
    {
        const uint64_t now = gpuNowNs(), maxAge = rbMaxAgeNs();
        uint64_t seq = 0u;
        size_t slot = 0u;
        {
            std::lock_guard<std::mutex> lock(s_rbMutex);
            const ReadbackEntry *best = nullptr;
            uint64_t bestScore = ~0ull, bestAge = 0u;
            int32_t bestDist = 0;
            for (const ReadbackEntry &e : s_rb)
            {
                if (!e.ready || e.seq == 0u || !rbSameKind(e.key, req) || e.data.size() != bytes || now - e.ns > maxAge)
                    continue;
                const int32_t dist = std::max(std::abs(e.key.x - req.x), std::abs(e.key.y - req.y));
                if (dist > kRbMaxDistance)
                    continue;
                // nearest first; among equally near ones the newest (2 pixels per 16 ms of age)
                const uint64_t score = static_cast<uint64_t>(dist) * 8000000ull + (now - e.ns);
                if (score < bestScore)
                {
                    bestScore = score;
                    best = &e;
                    bestAge = now - e.ns;
                    bestDist = dist;
                }
            }
            if (best)
            {
                std::memcpy(dst, best->data.data(), bytes);
                s_rbServed.fetch_add(1u, std::memory_order_relaxed);
                s_rbAgeNs.fetch_add(bestAge, std::memory_order_relaxed);
                s_rbDist.fetch_add(static_cast<uint64_t>(bestDist), std::memory_order_relaxed);
            }
            else if (asyncReadbackMode() >= 2 && (req.psm & 0x30u) == 0x30u)
            {
                std::memset(dst, 0xFF, bytes); // depth probe with no earlier read: covered
                s_rbCovered.fetch_add(1u, std::memory_order_relaxed);
            }
            else
            {
                s_rbMissed.fetch_add(1u, std::memory_order_relaxed);
                return false;
            }
            ReadbackEntry &e = rbNewEntry(req, now);
            seq = e.seq;
            slot = static_cast<size_t>(&e - s_rb);
        }
        // The read itself, in queue order behind the packet that set it up.
        PS2Memory::GpuJob job;
        const uint32_t pixelBytes = req.pixelBytes;
        job.fn = [slot, seq, bytes, pixelBytes]()
        {
            std::vector<uint8_t> buf(bytes, 0u);
            const uint32_t got = g_ps2GsReadbackHook ? g_ps2GsReadbackHook(buf.data(), bytes) : 0u;
            if (got < pixelBytes)
                return; // the transfer was not what the EE had set up (e.g. the packet is still masked)
            std::lock_guard<std::mutex> lock(s_rbMutex);
            ReadbackEntry &e = s_rb[slot];
            if (e.seq != seq)
                return; // slot reused meanwhile
            e.data = std::move(buf);
            e.ready = true;
            s_rbRefreshed.fetch_add(1u, std::memory_order_relaxed);
        };
        job.drain = false;
        gpuEnqueue(memory, std::move(job));
        return true;
    }
}

void PS2Memory::runGpuJob(GpuJob &job)
{
    for (const auto &packet : job.gif)
        submitGifPacket(GifPathId::Path3, packet.data(), static_cast<uint32_t>(packet.size()), false);
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto &stream : job.vif1)
        processVIF1Data(stream.data(), static_cast<uint32_t>(stream.size()));
    if (job.fn)
    {
        // Privileged register writes and flips take effect in order with the drawing: on the
        // GS thread when it has the drawing.
        if (t_onGpuWorker && g_gsStage)
        {
            g_gsStage->flush();
            GsStage::Item item;
            item.fn = std::move(job.fn);
            item.epoch = t_finishEpoch;
            g_gsStage->push(std::move(item));
        }
        else
            job.fn();
    }
    if (job.drain && m_gifArbiter)
        m_gifArbiter->drain();
    if (t_onGpuWorker && g_gsStage)
        g_gsStage->flush();
    if (job.countKick && !job.vif1.empty())
    {
        s_vif1Kicks.fetch_add(1u, std::memory_order_relaxed);
        s_vif1KickNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                               std::memory_order_relaxed);
    }
}

void vif1ObsReport(double frames)
{
    static uint64_t lastKicks = 0, lastKickNs = 0, lastBusy = 0, lastBp = 0, lastJobs = 0, lastLat = 0;
    static uint64_t lastGsBusy = 0, lastGsWait = 0, lastGsSync = 0, lastGsSyncs = 0;
    const uint64_t gsBusy = s_gsStageBusyNs.load(), gsWait = s_gsStageWaitNs.load(), gsSync = s_gsStageSyncNs.load(), gsSyncs = s_gsStageSyncs.load();
    static uint64_t lastSyncNs[PS2Memory::kGpuSyncReasonCount]{}, lastSyncCount[PS2Memory::kGpuSyncReasonCount]{};
    static const char *names[PS2Memory::kGpuSyncReasonCount] = {"dma", "privread", "csr", "vifreg", "vu1mem", "gscall", "other"};
    const uint64_t kicks = s_vif1Kicks.load(), kickNs = s_vif1KickNs.load(), busy = s_gpuBusyNs.load(),
                   bp = s_gpuBackpressureNs.load(), jobs = s_gpuJobs.load(), lat = s_gpuLatencyNs.load();
    std::string text;
    double syncMs = 0.0;
    for (uint32_t i = 0; i < PS2Memory::kGpuSyncReasonCount; ++i)
    {
        const uint64_t c = s_gpuSyncCount[i].load(), ns = s_gpuSyncNs[i].load();
        if (c != lastSyncCount[i])
        {
            char item[64];
            std::snprintf(item, sizeof(item), " %s %.1fx/%.2fms", names[i], (c - lastSyncCount[i]) / frames,
                          (ns - lastSyncNs[i]) / 1e6 / frames);
            text += item;
        }
        syncMs += (ns - lastSyncNs[i]) / 1e6 / frames;
        lastSyncCount[i] = c;
        lastSyncNs[i] = ns;
    }
    std::fprintf(stderr, "[ssx3:perf]   VIF1 kicks %.1f/frame, %.2f ms/frame | GPU worker: %s, busy %.1f ms/frame, %.1f jobs/frame, start latency %.2f ms/frame; EE waited %.2f ms/frame (syncs:%s), backpressure %.2f ms/frame\n",
                 (kicks - lastKicks) / frames, (kickNs - lastKickNs) / 1e6 / frames, gpuAsyncEnabled() ? "async" : "off",
                 (busy - lastBusy) / 1e6 / frames, (jobs - lastJobs) / frames, (lat - lastLat) / 1e6 / frames, syncMs, text.empty() ? " none" : text.c_str(),
                 (bp - lastBp) / 1e6 / frames);
    if (g_gsStage)
        std::fprintf(stderr, "[ssx3:perf]   GS thread: busy %.1f ms/frame; worker waited for it %.2f ms/frame (queue full) + %.1fx/%.2f ms (GS reads)\n",
                     (gsBusy - lastGsBusy) / 1e6 / frames, (gsWait - lastGsWait) / 1e6 / frames, (gsSyncs - lastGsSyncs) / frames,
                     (gsSync - lastGsSync) / 1e6 / frames);
    {
        static uint64_t lastServed = 0, lastMissed = 0, lastDirect = 0, lastRefreshed = 0, lastAge = 0, lastDist = 0, lastEarly = 0, lastDropped = 0, lastCovered = 0;
        const uint64_t served = s_rbServed.load(), missed = s_rbMissed.load(), direct = s_rbDirect.load(), refreshed = s_rbRefreshed.load(),
                       age = s_rbAgeNs.load(), dist = s_rbDist.load(), early = s_finishEarly.load(), dropped = s_finishDropped.load(),
                       covered = s_rbCovered.load();
        if (served != lastServed || direct != lastDirect || early != lastEarly || covered != lastCovered)
        {
            const uint64_t n = served - lastServed;
            std::fprintf(stderr, "[ssx3:perf]   GS read-backs per frame: %.1f answered from an earlier read (on average %.0f ms old, %.1f px away), %.2f with no earlier read answered as covered, %.2f read directly (%.2f of them waited: no earlier read); %.1f refreshed | FINISH reported early %.1f, queued FINISH dropped %.1f\n",
                         n / frames, n ? (age - lastAge) / 1e6 / n : 0.0, n ? static_cast<double>(dist - lastDist) / n : 0.0,
                         (covered - lastCovered) / frames, (direct - lastDirect) / frames, (missed - lastMissed) / frames,
                         (refreshed - lastRefreshed) / frames, (early - lastEarly) / frames, (dropped - lastDropped) / frames);
        }
        lastCovered = covered;
        lastServed = served;
        lastMissed = missed;
        lastDirect = direct;
        lastRefreshed = refreshed;
        lastAge = age;
        lastDist = dist;
        lastEarly = early;
        lastDropped = dropped;
    }
    lastGsBusy = gsBusy;
    lastGsWait = gsWait;
    lastGsSync = gsSync;
    lastGsSyncs = gsSyncs;
    lastKicks = kicks;
    lastKickNs = kickNs;
    lastBusy = busy;
    lastBp = bp;
    lastJobs = jobs;
    lastLat = lat;
}

namespace
{
    inline bool isGsPrivReg(uint32_t addr)
    {
        return Ps2AddressInRange(addr, PS2_GS_PRIV_REG_BASE, PS2_GS_PRIV_REG_SIZE);
    }

    inline bool isIoRegister(uint32_t addr)
    {
        return Ps2AddressInRange(addr, PS2_IO_BASE, PS2_IO_SIZE);
    }

    inline uint64_t *gsRegPtr(GSRegisters &gs, uint32_t addr)
    {
        // Support both 64-bit base offsets and +4 dword aliases.
        uint32_t off = (addr - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        switch (off)
        {
        case 0x0000:
            return &gs.pmode;
        case 0x0010:
            return &gs.smode1;
        case 0x0020:
            return &gs.smode2;
        case 0x0030:
            return &gs.srfsh;
        case 0x0040:
            return &gs.synch1;
        case 0x0050:
            return &gs.synch2;
        case 0x0060:
            return &gs.syncv;
        case 0x0070:
            return &gs.dispfb1;
        case 0x0080:
            return &gs.display1;
        case 0x0090:
            return &gs.dispfb2;
        case 0x00A0:
            return &gs.display2;
        case 0x00B0:
            return &gs.extbuf;
        case 0x00C0:
            return &gs.extdata;
        case 0x00D0:
            return &gs.extwrite;
        case 0x00E0:
            return &gs.bgcolor;
        // CSR (offset 0x1000) is intentionally not handled here: it is
        // std::atomic<uint64_t> and no longer converts to uint64_t*. Callers must
        // check for offset 0x1000 themselves and go through writeCsrHalf/
        // writeCsrFull/gs.csr.load() instead of gsRegPtr().
        case 0x1010:
            return &gs.imr;
        case 0x1040:
            return &gs.busdir;
        case 0x1080:
            return &gs.siglblid;
        default:
            return nullptr;
        }
    }

    constexpr uint32_t kGsCsrRegOffset = 0x1000u;

    // CSR bits 14-15 are the GIF->GS FIFO status (00 = neither, 01 = empty, 10 = almost full).
    // This runtime consumes GIF packets synchronously, so from the EE's point of view the FIFO
    // is always drained. Games (e.g. SSX 3) spin on "(CSR & 0xC000) == 0x4000" before touching
    // the GS, which never completes if these bits read back as 00.
    inline uint64_t gsCsrReadValue(uint64_t raw)
    {
        return (raw & ~0xC000ull) | 0x4000ull;
    }

    // Atomically apply a 32-bit write to one half (off=0 low dword, off=4 high
    // dword) of the GS CSR register. Bits 0..1 of the low dword (SIGNAL/FINISH) are
    // write-one-to-clear; everything else is a plain merge. Uses compare_exchange
    // so the whole read-modify-write is a single atomic step -- this register is
    // also touched by the vsync worker (FIELD bit) and the GIF (SIGNAL/FINISH) on
    // other threads, so a load-then-store here would race with them.
    inline void writeCsrHalf(std::atomic<uint64_t> &csr, uint32_t off, uint32_t value)
    {
        constexpr uint32_t kW1cMask = 0x3u;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            if (off == 0u)
            {
                uint32_t oldLow = static_cast<uint32_t>(expected & 0xFFFFFFFFull);
                uint32_t mergedLow = (oldLow & kW1cMask) | (value & ~kW1cMask);
                desired = (expected & 0xFFFFFFFF00000000ull) | static_cast<uint64_t>(mergedLow);
                desired &= ~static_cast<uint64_t>(value & kW1cMask);
            }
            else
            {
                uint64_t mask = 0xFFFFFFFFull << (off * 8u);
                desired = (expected & ~mask) | (static_cast<uint64_t>(value) << (off * 8u));
            }
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    // Same as writeCsrHalf but for a full 64-bit CSR write (bits 0..1 are still
    // write-one-to-clear against the current value).
    inline void writeCsrFull(std::atomic<uint64_t> &csr, uint64_t value)
    {
        constexpr uint64_t kW1cMask = 0x3ull;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            desired = (expected & kW1cMask) | (value & ~kW1cMask);
            desired &= ~(value & kW1cMask);
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    constexpr std::array<uint32_t, 4> kEeTimerBases = {
        0x10000000u,
        0x10000800u,
        0x10001000u,
        0x10001800u,
    };
    constexpr uint32_t kEeTimerCountOffset = 0x00u;
    constexpr uint32_t kEeTimerModeOffset = 0x10u;
    constexpr uint32_t kEeTimerCompareOffset = 0x20u;
    constexpr uint32_t kEeTimerHoldOffset = 0x30u;
    constexpr uint32_t kEeTimerModeClksMask = 0x3u;
    constexpr uint32_t kEeTimerModeConfigMask = 0x3FFu;
    constexpr uint32_t kEeTimerModeStatusMask = 0xC00u;
    constexpr uint32_t kEeTimerModeZret = 1u << 6;
    constexpr uint32_t kEeTimerModeCue = 1u << 7;
    constexpr uint32_t kEeTimerModeCmpe = 1u << 8;
    constexpr uint32_t kEeTimerModeOvfe = 1u << 9;
    constexpr uint32_t kEeTimerModeEquf = 1u << 10;
    constexpr uint32_t kEeTimerModeOvff = 1u << 11;
    constexpr uint64_t kEeClockHz = 294912000ull;
    constexpr std::array<uint64_t, 4> kEeTimerClockHz = {
        147456000ull,
        9216000ull,
        576000ull,
        15734ull,
    };

    inline bool decodeEeTimerRegister(uint32_t address, size_t &timerIndex, uint32_t &offset)
    {
        for (size_t index = 0; index < kEeTimerBases.size(); ++index)
        {
            const uint32_t candidateOffset = address - kEeTimerBases[index];
            if (candidateOffset == kEeTimerCountOffset ||
                candidateOffset == kEeTimerModeOffset ||
                candidateOffset == kEeTimerCompareOffset ||
                (index < 2u && candidateOffset == kEeTimerHoldOffset))
            {
                timerIndex = index;
                offset = candidateOffset;
                return true;
            }
        }
        return false;
    }

    constexpr uint64_t ticksUntilMatch(uint32_t count, uint32_t target)
    {
        const uint32_t distance = (target - count) & 0xFFFFu;
        return distance == 0u ? 0x10000ull : static_cast<uint64_t>(distance);
    }

    struct DmaTagView
    {
        uint16_t qwc = 0;
        uint8_t id = 0;
        bool irq = false;
        uint32_t addr = 0;
        uint32_t upper = 0;
    };

    inline DmaTagView decodeDmaTag(uint64_t tag)
    {
        DmaTagView out{};
        out.qwc = static_cast<uint16_t>(tag & 0xFFFFu);
        out.id = static_cast<uint8_t>((tag >> 28u) & 0x7u);
        out.irq = ((tag >> 31u) & 0x1ull) != 0ull;
        out.addr = static_cast<uint32_t>((tag >> 32u) & 0x7FFFFFFFu);
        out.upper = static_cast<uint32_t>((tag >> 16u) & 0xFFFFu);
        return out;
    }

    inline uint32_t gifTagNloop(uint64_t tagLo)
    {
        return static_cast<uint32_t>(tagLo & 0x7FFFu);
    }

    inline uint8_t gifTagFlg(uint64_t tagLo)
    {
        return static_cast<uint8_t>((tagLo >> 58u) & 0x3u);
    }

    inline uint32_t gifTagNreg(uint64_t tagLo)
    {
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60u) & 0xFu);
        return nreg == 0u ? 16u : nreg;
    }

}

// Helpers for GS VRAM addressing (PSMCT32 path).
static inline uint32_t gs_vram_offset(uint32_t basePage, uint32_t x, uint32_t y, uint32_t fbw)
{
    // basePage is in 2048-byte units; fbw is in blocks of 64 pixels.
    uint32_t strideBytes = fbw * 64 * 4;
    return basePage * 2048 + y * strideBytes + x * 4;
}

PS2Memory::PS2Memory()
    : m_rdram(nullptr), m_scratchpad(nullptr), iop_ram(nullptr), m_seenGifCopy(false), m_gsVRAM(nullptr)
{
    ps2SetScratchpadHostPtr(nullptr);
}

PS2Memory::~PS2Memory()
{
    if (m_rdram)
    {
        delete[] m_rdram;
        m_rdram = nullptr;
    }

    if (m_scratchpad)
    {
        ps2SetScratchpadHostPtr(nullptr);
        delete[] m_scratchpad;
        m_scratchpad = nullptr;
    }

    if (m_gsVRAM)
    {
        delete[] m_gsVRAM;
        m_gsVRAM = nullptr;
    }

    if (m_vu1Code)
    {
        delete[] m_vu1Code;
        m_vu1Code = nullptr;
    }
    if (m_vu1Data)
    {
        delete[] m_vu1Data;
        m_vu1Data = nullptr;
    }
    if (m_vu0Code)
    {
        delete[] m_vu0Code;
        m_vu0Code = nullptr;
    }
    if (m_vu0Data)
    {
        delete[] m_vu0Data;
        m_vu0Data = nullptr;
    }

    if (iop_ram)
    {
        delete[] iop_ram;
        iop_ram = nullptr;
    }
}

bool PS2Memory::initialize(size_t ramSize)
{
    auto cleanup = [this]()
    {
        delete[] m_rdram;
        delete[] m_scratchpad;
        delete[] iop_ram;
        delete[] m_gsVRAM;
        delete[] m_vu0Code;
        delete[] m_vu0Data;
        delete[] m_vu1Code;
        delete[] m_vu1Data;
        m_rdram = nullptr;
        m_scratchpad = nullptr;
        ps2SetScratchpadHostPtr(nullptr);
        iop_ram = nullptr;
        m_gsVRAM = nullptr;
        m_vu0Code = nullptr;
        m_vu0Data = nullptr;
        m_vu1Code = nullptr;
        m_vu1Data = nullptr;
    };

    cleanup();
    m_seenGifCopy = false;
    m_dmaStartCount.store(0, std::memory_order_relaxed);
    m_gifCopyCount.store(0, std::memory_order_relaxed);
    m_gsWriteCount.store(0, std::memory_order_relaxed);
    m_vifWriteCount.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_completedDmacMutex);
        m_completedDmacCauses.clear();
    }
    m_codeRegions.clear();
    m_path3Masked = false;
    m_path3MaskedFifo.clear();
    m_vif1PendingPath2ImageQwc = 0u;
    m_vif1PendingPath2DirectHl = false;
    resetEeTimers();

    try
    {
        // Allocate main RAM
        m_rdram = new uint8_t[ramSize];
        std::memset(m_rdram, 0, ramSize);

        // Allocate scratchpad
        m_scratchpad = new uint8_t[PS2_SCRATCHPAD_SIZE];
        std::memset(m_scratchpad, 0, PS2_SCRATCHPAD_SIZE);
        ps2SetScratchpadHostPtr(m_scratchpad);

        // Initialize EE TLB entries (R5900 has 48 entries).
        m_tlbEntries.assign(48, TLBEntry{0, 0, 0, false});

        // Allocate IOP RAM
        iop_ram = new uint8_t[2 * 1024 * 1024]; // 2MB

        // Initialize IOP RAM with zeros
        std::memset(iop_ram, 0, 2 * 1024 * 1024);

        // Initialize I/O registers
        m_ioRegisters.clear();

        // Initialize GS registers
        memset(&gs_regs, 0, sizeof(gs_regs));
        // memset zero-fills std::atomic<uint64_t>::csr's bytes, which is not itself
        // a guaranteed-valid atomic store; make the zero-initialization explicit.
        gs_regs.csr.store(0);
        gs_regs.dispfb1 = (0ULL << 0) | (10ULL << 9) | (0ULL << 15) | (0ULL << 32) | (0ULL << 43);
        gs_regs.display1 = (0ULL << 0) | (0ULL << 12) | (0ULL << 23) | (0ULL << 27) | (639ULL << 32) | (447ULL << 44);
        gs_regs.dispfb2 = gs_regs.dispfb1;
        gs_regs.display2 = gs_regs.display1;

        // Allocate GS VRAM (4MB)
        m_gsVRAM = new uint8_t[PS2_GS_VRAM_SIZE];
        std::memset(m_gsVRAM, 0, PS2_GS_VRAM_SIZE);

        m_vu0Code = new uint8_t[PS2_VU0_CODE_SIZE];
        m_vu0Data = new uint8_t[PS2_VU0_DATA_SIZE];
        std::memset(m_vu0Code, 0, PS2_VU0_CODE_SIZE);
        std::memset(m_vu0Data, 0, PS2_VU0_DATA_SIZE);

        m_vu1Code = new uint8_t[PS2_VU1_CODE_SIZE];
        m_vu1Data = new uint8_t[PS2_VU1_DATA_SIZE];
        std::memset(m_vu1Code, 0, PS2_VU1_CODE_SIZE);
        std::memset(m_vu1Data, 0, PS2_VU1_DATA_SIZE);
        markVU0CodeModified();
        markVU1CodeModified();

        // Initialize VIF registers
        memset(&vif0_regs, 0, sizeof(vif0_regs));
        memset(&vif1_regs, 0, sizeof(vif1_regs));

        // Initialize DMA registers
        memset(dma_regs, 0, sizeof(dma_regs));

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error initializing PS2 memory: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void PS2Memory::resetEeTimers() noexcept
{
    m_eeTimers = {};
}

uint32_t PS2Memory::advanceEeTimers(uint64_t eeCycles) noexcept
{
    if (eeCycles == 0u)
    {
        return 0u;
    }

    constexpr uint32_t kGifStat = 0x10003020u;
    constexpr uint32_t kGifFqcMask = 0x1F000000u;
    auto gifStatIt = m_ioRegisters.find(kGifStat);
    if (gifStatIt != m_ioRegisters.end())
        gifStatIt->second &= ~kGifFqcMask;

    uint32_t interruptMask = 0u;
    for (size_t index = 0; index < m_eeTimers.size(); ++index)
    {
        EeTimer &timer = m_eeTimers[index];
        if ((timer.mode & kEeTimerModeCue) == 0u)
        {
            continue;
        }

        const uint64_t clockHz = kEeTimerClockHz[timer.mode & kEeTimerModeClksMask];
        const uint64_t wholeSeconds = eeCycles / kEeClockHz;
        const uint64_t remainingCycles = eeCycles % kEeClockHz;
        const uint64_t scaled = remainingCycles * clockHz + timer.clockRemainder;
        const uint64_t ticks = wholeSeconds * clockHz + scaled / kEeClockHz;
        timer.clockRemainder = scaled % kEeClockHz;
        if (ticks == 0u)
        {
            continue;
        }

        const uint32_t oldCount = timer.count & 0xFFFFu;
        const uint32_t compare = timer.compare & 0xFFFFu;
        const uint64_t compareDistance = ticksUntilMatch(oldCount, compare);
        const uint64_t overflowDistance = 0x10000ull - oldCount;
        const bool zeroReturn = (timer.mode & kEeTimerModeZret) != 0u;
        const bool compareReached = ticks >= compareDistance;
        bool overflowReached = false;

        if (zeroReturn)
        {
            overflowReached = ticks >= overflowDistance && overflowDistance <= compareDistance;
            if (compareReached)
            {
                const uint64_t remaining = ticks - compareDistance;
                timer.count = compare == 0u
                                  ? static_cast<uint32_t>(remaining & 0xFFFFu)
                                  : static_cast<uint32_t>(remaining % compare);
            }
            else
            {
                timer.count = static_cast<uint32_t>((oldCount + ticks) & 0xFFFFu);
            }
        }
        else
        {
            overflowReached = ticks >= overflowDistance;
            timer.count = static_cast<uint32_t>((oldCount + ticks) & 0xFFFFu);
        }

        if (compareReached && (timer.mode & kEeTimerModeCmpe) != 0u && (timer.mode & kEeTimerModeEquf) == 0u)
        {
            timer.mode |= kEeTimerModeEquf;
            interruptMask |= 1u << index;
        }
        if (overflowReached && (timer.mode & kEeTimerModeOvfe) != 0u && (timer.mode & kEeTimerModeOvff) == 0u)
        {
            timer.mode |= kEeTimerModeOvff;
            interruptMask |= 1u << index;
        }
    }
    return interruptMask;
}

uint64_t PS2Memory::cyclesUntilNextEeTimerInterrupt() const noexcept
{
    uint64_t nearest = std::numeric_limits<uint64_t>::max();
    for (const EeTimer &timer : m_eeTimers)
    {
        if ((timer.mode & kEeTimerModeCue) == 0u)
        {
            continue;
        }

        const uint32_t count = timer.count & 0xFFFFu;
        const uint32_t compare = timer.compare & 0xFFFFu;
        const uint64_t compareDistance = ticksUntilMatch(count, compare);
        const uint64_t overflowDistance = 0x10000ull - count;
        uint64_t eventTicks = std::numeric_limits<uint64_t>::max();

        if ((timer.mode & kEeTimerModeCmpe) != 0u &&
            (timer.mode & kEeTimerModeEquf) == 0u)
        {
            eventTicks = compareDistance;
        }
        const bool overflowCanOccur = (timer.mode & kEeTimerModeZret) == 0u ||
                                      overflowDistance <= compareDistance;
        if (overflowCanOccur &&
            (timer.mode & kEeTimerModeOvfe) != 0u &&
            (timer.mode & kEeTimerModeOvff) == 0u)
        {
            eventTicks = std::min(eventTicks, overflowDistance);
        }
        if (eventTicks == std::numeric_limits<uint64_t>::max())
        {
            continue;
        }

        const uint64_t clockHz = kEeTimerClockHz[timer.mode & kEeTimerModeClksMask];
        const uint64_t numerator = eventTicks * kEeClockHz - timer.clockRemainder;
        const uint64_t cycles = (numerator + clockHz - 1u) / clockHz;
        nearest = std::min(nearest, std::max<uint64_t>(1u, cycles));
    }
    return nearest;
}

bool PS2Memory::isScratchpad(uint32_t address) const
{
    return ps2IsScratchpadAddress(address);
}

uint8_t *PS2Memory::mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit)
{
    return const_cast<uint8_t *>(static_cast<const PS2Memory *>(this)->mapVuMemory(physAddr, size, offset, limit));
}

const uint8_t *PS2Memory::mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit) const
{
    if (physAddr >= PS2_VU1_CODE_BASE && physAddr < PS2_VU1_DATA_BASE + PS2_VU1_DATA_SIZE && gpuBusy())
        gpuSyncGlobal(kGpuSyncVu1Mem); // VU1 memory belongs to the GPU worker while it runs
    auto mapRange = [&](uint32_t base, uint32_t rangeSize, const uint8_t *ptr) -> const uint8_t *
    {
        if (!ptr || physAddr < base)
        {
            return nullptr;
        }
        const uint32_t local = physAddr - base;
        if (local >= rangeSize || size > (rangeSize - local))
        {
            return nullptr;
        }
        offset = local;
        limit = rangeSize;
        return ptr;
    };

    if (const uint8_t *ptr = mapRange(PS2_VU0_CODE_BASE, PS2_VU0_CODE_SIZE, m_vu0Code))
    {
        return ptr;
    }
    if (const uint8_t *ptr = mapRange(PS2_VU0_DATA_BASE, PS2_VU0_DATA_SIZE, m_vu0Data))
    {
        return ptr;
    }
    if (const uint8_t *ptr = mapRange(PS2_VU1_CODE_BASE, PS2_VU1_CODE_SIZE, m_vu1Code))
    {
        return ptr;
    }
    return mapRange(PS2_VU1_DATA_BASE, PS2_VU1_DATA_SIZE, m_vu1Data);
}

uint32_t PS2Memory::translateAddress(uint32_t virtualAddress)
{
    if (isScratchpad(virtualAddress))
    {
        return ps2ScratchpadOffset(virtualAddress);
    }

    // EE uncached aliases of main RAM (per PS2 memory map):
    //   0x20000000-0x3FFFFFFF -> 32MB mirror of RDRAM
    // This includes the accelerated window rooted at 0x30100000.
    if (Ps2IsUncachedRamMirrorAddress(virtualAddress))
    {
        return virtualAddress & PS2_RAM_MASK;
    }

    // KSEG0/KSEG1 direct-mapped window.
    if (Ps2IsKseg01Address(virtualAddress))
    {
        return Ps2DirectMappedPhysicalAddress(virtualAddress);
    }

    // In this runtime, low segments are treated as physical-style addresses already.
    if (virtualAddress < 0x80000000)
    {
        return virtualAddress;
    }

    // KSEG2/KSEG3 are TLB mapped.
    if (Ps2IsKseg23Address(virtualAddress))
    {
        for (const auto &entry : m_tlbEntries)
        {
            if (entry.valid)
            {
                // PageMask uses bits [24:13]. Build an address-level mask (plus 4KB base page bits).
                const uint32_t mask = entry.mask & 0x01FFE000u;
                const uint32_t compareMask = ~(mask | 0xFFFu);
                if ((virtualAddress & compareMask) == (entry.vpn & compareMask))
                {
                    // TLB hit
                    const uint32_t pageOffsetMask = mask | 0xFFFu;
                    const uint32_t physBase = entry.pfn << 12;
                    return physBase | (virtualAddress & pageOffsetMask);
                }
            }
        }
        throw std::runtime_error("TLB miss for address: 0x" + std::to_string(virtualAddress));
    }

    return virtualAddress;
}

bool PS2Memory::tlbRead(uint32_t index, uint32_t &vpn, uint32_t &pfn, uint32_t &mask, bool &valid) const
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    const TLBEntry &entry = m_tlbEntries[index];
    vpn = entry.vpn;
    pfn = entry.pfn;
    mask = entry.mask;
    valid = entry.valid;
    return true;
}

bool PS2Memory::tlbWrite(uint32_t index, uint32_t vpn, uint32_t pfn, uint32_t mask, bool valid)
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    TLBEntry &entry = m_tlbEntries[index];
    entry.vpn = vpn & 0xFFFFF000u;
    entry.pfn = pfn & 0x000FFFFFu;
    entry.mask = mask & 0x01FFE000u;
    entry.valid = valid;
    return true;
}

int32_t PS2Memory::tlbProbe(uint32_t vpn) const
{
    const uint32_t normalizedVpn = vpn & 0xFFFFF000u;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_tlbEntries.size()); ++i)
    {
        const TLBEntry &entry = m_tlbEntries[i];
        if (!entry.valid)
        {
            continue;
        }

        const uint32_t mask = entry.mask & 0x01FFE000u;
        const uint32_t compareMask = ~(mask | 0xFFFu);
        if ((normalizedVpn & compareMask) == (entry.vpn & compareMask))
        {
            return static_cast<int32_t>(i);
        }
    }

    return -1;
}

uint8_t PS2Memory::read8(uint32_t address)
{
    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return m_scratchpad[physAddr];
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return m_rdram[physAddr];
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint8_t), vuOffset, vuLimit))
    {
        (void)vuLimit;
        return vuMem[vuOffset];
    }
    else if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t value = readIORegister(regAddr);
        uint32_t shift = (physAddr & 3) * 8;
        return static_cast<uint8_t>((value >> shift) & 0xFF);
    }

    return 0;
}

uint16_t PS2Memory::read16(uint32_t address)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit read at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint16_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read16 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint16_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read16 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint16_t), vuOffset, vuLimit))
    {
        return loadScalar<uint16_t>(vuMem, vuOffset, vuLimit, "read16 vu", address);
    }
    else if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t value = readIORegister(regAddr);
        uint32_t shift = (physAddr & 2) * 8;
        return static_cast<uint16_t>((value >> shift) & 0xFFFF);
    }

    return 0;
}

uint32_t PS2Memory::read32(uint32_t address)
{
    if (address & 3)
    {
        throw std::runtime_error("Unaligned 32-bit read at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        uint32_t off = address & 7;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            uint64_t val = gsCsrReadValue(gs_regs.csr.load());
            return (uint32_t)(val >> (off * 8));
        }
        if (gpuBusy())
            gpuSyncGlobal(kGpuSyncPrivRead);
        uint64_t *reg = gsRegPtr(gs_regs, address);
        if (!reg)
            return 0;
        uint64_t val = *reg;
        return (uint32_t)(val >> (off * 8));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint32_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read32 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint32_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read32 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint32_t), vuOffset, vuLimit))
    {
        return loadScalar<uint32_t>(vuMem, vuOffset, vuLimit, "read32 vu", address);
    }
    else if (isIoRegister(physAddr))
    {
        return readIORegister(physAddr);
    }

    return 0;
}

uint64_t PS2Memory::read64(uint32_t address)
{
    if (address & 7)
    {
        throw std::runtime_error("Unaligned 64-bit read at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            return gsCsrReadValue(gs_regs.csr.load());
        }
        if (gpuBusy())
            gpuSyncGlobal(kGpuSyncPrivRead);
        uint64_t *reg = gsRegPtr(gs_regs, address);
        return reg ? *reg : 0;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint64_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read64 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint64_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read64 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint64_t), vuOffset, vuLimit))
    {
        return loadScalar<uint64_t>(vuMem, vuOffset, vuLimit, "read64 vu", address);
    }

    // 64-bit IO read: compose from the two adjacent 32-bit IO register slots
    // to avoid any side-effects from read32 handlers.
    if (isIoRegister(address))
    {
        uint32_t lo = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t hi = m_ioRegisters.count(address + 4) ? m_ioRegisters[address + 4] : 0u;
        return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }
    return (uint64_t)read32(address) | ((uint64_t)read32(address + 4) << 32);
}

__m128i PS2Memory::read128(uint32_t address)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit read at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        inRange(physAddr, sizeof(__m128i), PS2_SCRATCHPAD_SIZE, "read128 scratchpad", address);
        return _mm_loadu_si128(reinterpret_cast<__m128i *>(&m_scratchpad[physAddr]));
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        inRange(physAddr, sizeof(__m128i), PS2_RAM_SIZE, "read128 rdram", address);
        return _mm_loadu_si128(reinterpret_cast<__m128i *>(&m_rdram[physAddr]));
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(__m128i), vuOffset, vuLimit))
    {
        inRange(vuOffset, sizeof(__m128i), vuLimit, "read128 vu", address);
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(vuMem + vuOffset));
    }

    // 128-bit reads are primarily for quad-word loads in the EE, which are only valid for RAM areas
    // Return zeroes for unsupported areas
    return _mm_setzero_si128();
}

void PS2Memory::write8(uint32_t address, uint8_t value)
{
    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        m_scratchpad[physAddr] = value;
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        m_rdram[physAddr] = value;
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint8_t), vuOffset, vuLimit))
        {
            (void)vuLimit;
            vuMem[vuOffset] = value;
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        // IO registers - handle byte writes by modifying the appropriate byte in the word
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t shift = (physAddr & 3) * 8;
        uint32_t mask = ~(0xFF << shift);
        uint32_t newValue = (m_ioRegisters[regAddr] & mask) | ((uint32_t)value << shift);
        writeIORegister(regAddr, newValue);
    }
}

void PS2Memory::write16(uint32_t address, uint16_t value)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit write at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint16_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write16 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        storeScalar<uint16_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write16 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint16_t), vuOffset, vuLimit))
        {
            storeScalar<uint16_t>(vuMem, vuOffset, vuLimit, value, "write16 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t shift = (physAddr & 2) * 8;
        uint32_t mask = ~(0xFFFF << shift);
        uint32_t newValue = (m_ioRegisters[regAddr] & mask) | ((uint32_t)value << shift);
        writeIORegister(regAddr, newValue);
    }
}

void PS2Memory::write32(uint32_t address, uint32_t value)
{
    if (address & 3)
    {
        throw std::runtime_error("Unaligned 32-bit write at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        uint32_t off = address & 7;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            // CSR: bits 0..1 of the low dword are write-one-to-clear status bits.
            // Done as a single atomic RMW -- see writeCsrHalf's comment.
            // Clearing SIGNAL or resetting the GS waits for the queued GS work first; clearing
            // only FINISH starts a new FINISH epoch instead (see s_finishEpoch).
            std::unique_lock<std::mutex> finishLock;
            if (off == 0u && (value & 0x203u) != 0u)
            {
                if ((value & 0x203u) == 0x2u && asyncReadbackEnabled())
                {
                    finishLock = std::unique_lock<std::mutex>(s_finishMutex);
                    s_finishEpoch.fetch_add(1u, std::memory_order_release);
                }
                else if (gpuBusy())
                    gpuSyncGlobal(kGpuSyncCsr);
            }
            writeCsrHalf(gs_regs.csr, off, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const bool flip = (regOff == 0x0070u || regOff == 0x0090u) && off == 4u; // DISPFBn high word completes it
            auto apply = [reg, off, value, flip]()
            {
                uint64_t mask = 0xFFFFFFFFULL << (off * 8);
                *reg = (*reg & ~mask) | ((uint64_t)value << (off * 8));
                if (flip && g_ps2FlipHook)
                    g_ps2FlipHook();
            };
            if (gpuBusy())
            {
                // Display registers (DISPFB etc.) take effect in order with the queued drawing.
                GpuJob job;
                job.fn = apply;
                job.drain = false;
                gpuEnqueue(this, std::move(job));
            }
            else
                apply();
        }
        return;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint32_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write32 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        // Check if this might be code modification
        markModified(address, 4);

        storeScalar<uint32_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write32 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint32_t), vuOffset, vuLimit))
        {
            storeScalar<uint32_t>(vuMem, vuOffset, vuLimit, value, "write32 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        writeIORegister(physAddr, value);
    }
}

void PS2Memory::write64(uint32_t address, uint64_t value)
{
    if (address & 7)
    {
        throw std::runtime_error("Unaligned 64-bit write at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            // CSR: bits 0..1 are write-one-to-clear status bits. Done as a single
            // atomic RMW -- see writeCsrFull's comment.
            std::unique_lock<std::mutex> finishLock;
            if ((value & 0x203u) != 0u)
            {
                if ((value & 0x203u) == 0x2u && asyncReadbackEnabled())
                {
                    finishLock = std::unique_lock<std::mutex>(s_finishMutex);
                    s_finishEpoch.fetch_add(1u, std::memory_order_release);
                }
                else if (gpuBusy())
                    gpuSyncGlobal(kGpuSyncCsr);
            }
            writeCsrFull(gs_regs.csr, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const bool flip = regOff == 0x0070u || regOff == 0x0090u;
            auto apply = [reg, value, flip]()
            {
                *reg = value;
                if (flip && g_ps2FlipHook)
                    g_ps2FlipHook();
            };
            if (gpuBusy())
            {
                GpuJob job;
                job.fn = apply;
                job.drain = false;
                gpuEnqueue(this, std::move(job));
            }
            else
                apply();
        }
        return;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint64_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write64 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(address, 8);
        storeScalar<uint64_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write64 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint64_t), vuOffset, vuLimit))
        {
            storeScalar<uint64_t>(vuMem, vuOffset, vuLimit, value, "write64 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        write32(address, (uint32_t)value);
        write32(address + 4, (uint32_t)(value >> 32));
    }
}

void PS2Memory::write128(uint32_t address, __m128i value)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit write at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (!scratch && physAddr == 0x10004000u) // VIF0_FIFO
    {
        alignas(16) uint8_t fifoData[16];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(fifoData), value);
        processVIF0Data(fifoData, sizeof(fifoData));
        return;
    }
    if (!scratch && physAddr == 0x10005000u) // VIF1_FIFO
    {
        alignas(16) uint8_t fifoData[16];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(fifoData), value);
        if (gpuAsyncEnabled())
        {
            GpuJob job;
            job.vif1.emplace_back(fifoData, fifoData + sizeof(fifoData));
            job.drain = false;
            gpuEnqueue(this, std::move(job));
        }
        else
            processVIF1Data(fifoData, sizeof(fifoData));
        return;
    }

    if (scratch)
    {
        inRange(physAddr, sizeof(__m128i), PS2_SCRATCHPAD_SIZE, "write128 scratchpad", address);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&m_scratchpad[physAddr]), value);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(address, 16);
        inRange(physAddr, sizeof(__m128i), PS2_RAM_SIZE, "write128 rdram", address);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&m_rdram[physAddr]), value);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(__m128i), vuOffset, vuLimit))
        {
            inRange(vuOffset, sizeof(__m128i), vuLimit, "write128 vu", address);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(vuMem + vuOffset), value);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        // Non-RAM 128-bit stores are modeled as two 64-bit stores.
        uint64_t lo = _mm_extract_epi64(value, 0);
        uint64_t hi = _mm_extract_epi64(value, 1);

        write64(address, lo);
        write64(address + 8, hi);
    }
}

bool PS2Memory::writeIORegister(uint32_t address, uint32_t value)
{
    size_t timerIndex = 0u;
    uint32_t timerOffset = 0u;
    if (decodeEeTimerRegister(address, timerIndex, timerOffset))
    {
        EeTimer &timer = m_eeTimers[timerIndex];
        switch (timerOffset)
        {
        case kEeTimerCountOffset:
            timer.count = value & 0xFFFFu;
            timer.clockRemainder = 0u;
            break;
        case kEeTimerModeOffset:
        {
            const uint32_t previousMode = timer.mode;
            const uint32_t status = (previousMode & kEeTimerModeStatusMask) & ~(value & kEeTimerModeStatusMask);
            timer.mode = (value & kEeTimerModeConfigMask) | status;
            if (((previousMode ^ timer.mode) & (kEeTimerModeClksMask | kEeTimerModeCue)) != 0u)
            {
                timer.clockRemainder = 0u;
            }
            break;
        }
        case kEeTimerCompareOffset:
            timer.compare = value & 0xFFFFu;
            break;
        case kEeTimerHoldOffset:
            timer.hold = value & 0xFFFFu;
            break;
        default:
            return false;
        }
        return true;
    }

    if (isGsPrivReg(address))
    {
        // NB: unreachable from write8/16/32/64 today since those all funnel IO
        // register writes through addresses in PS2_IO_BASE's range, which is
        // disjoint from PS2_GS_PRIV_REG_BASE; kept correct for direct callers.
        m_ioRegisters[address] = value;
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            writeCsrHalf(gs_regs.csr, off, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const uint64_t mask = 0xFFFFFFFFull << (off * 8u);
            *reg = (*reg & ~mask) | (static_cast<uint64_t>(value) << (off * 8u));
        }
        m_gsWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        if (address == 0x10002010)
        {
            m_ioRegisters[address] = value & ~(1u << 31);
            if (value & (1u << 30))
            {
                m_ioRegisters[0x10002000] = 0;
                m_ioRegisters[0x10002020] = 0;
                m_ioRegisters[0x10002030] = 0;
            }
        }
        else
        {
            m_ioRegisters[address] = value;
        }
        return true;
    }

    if (address == 0x1000E010u)
    {
        const uint32_t current = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t status = current & 0x3FFu;
        uint32_t mask = (current >> 16) & 0x3FFu;

        // D_STAT low bits are W1C status, high bits [16..25] toggle masks on write-one.
        status &= ~(value & 0x3FFu);
        mask ^= ((value >> 16) & 0x3FFu);

        uint32_t next = (current & ~((0x3FFu) | (0x3FFu << 16) | (1u << 31)));
        next |= status | (mask << 16);
        if ((status & mask) != 0u)
            next |= (1u << 31);
        m_ioRegisters[address] = next;
        return true;
    }

    // VIF1 state belongs to the GPU worker while it runs. A VIF1_STAT write only sets FDR (the
    // FIFO direction, for GS downloads), which the worker does not use: no wait for that one.
    if (address >= 0x10003C00u && address < 0x10003E00u && gpuBusy() && !(address == 0x10003C00u && asyncReadbackEnabled()))
        gpuSyncGlobal(kGpuSyncVifReg);

    m_ioRegisters[address] = value;

    if (address >= 0x10003C00u && address < 0x10003E00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);

        switch (address)
        {
        case 0x10003C10u:     // VIF1_FBRST
            if (value & 0x1u) // RST
            {
                const bool wasPath3Masked = m_path3Masked;
                std::memset(&vif1_regs, 0, sizeof(vif1_regs));
                m_vif1PendingPath2ImageQwc = 0u;
                m_vif1PendingPath2DirectHl = false;
                m_path3Masked = false;
                if (wasPath3Masked)
                    flushMaskedPath3Packets();
            }
            if (value & 0x8u) // STC
            {
                vif1_regs.stat &= ~((1u << 8) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12) | (1u << 13));
            }
            break;
        case 0x10003C30u:
            vif1_regs.mark = value & 0xFFFFu;
            vif1_regs.stat &= ~(1u << 6); // clear MRK flag on CPU write
            break;
        case 0x10003C40u:
            vif1_regs.cycle = value & 0xFFFFu;
            break;
        case 0x10003C50u:
            vif1_regs.mode = value & 0x3u;
            break;
        case 0x10003C60u:
            vif1_regs.num = value & 0xFFu;
            break;
        case 0x10003C70u:
            vif1_regs.mask = value;
            break;
        case 0x10003C80u:
            vif1_regs.code = value;
            break;
        case 0x10003C90u:
            vif1_regs.itops = value & 0x3FFu;
            break;
        case 0x10003CA0u:
            vif1_regs.base = value & 0x3FFu;
            break;
        case 0x10003CB0u:
            vif1_regs.ofst = value & 0x3FFu;
            break;
        case 0x10003CC0u:
            vif1_regs.tops = value & 0x3FFu;
            break;
        case 0x10003CD0u:
            vif1_regs.itop = value & 0x3FFu;
            break;
        case 0x10003CE0u:
            vif1_regs.top = value & 0x3FFu;
            break;
        default:
            break;
        }

        return true;
    }

    if (address >= 0x10003800u && address < 0x10003A00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10008000 && address < 0x1000F000)
    {
        if ((address & 0xFF) == 0x00 && (value & 0x100))
        {
            const auto dctrlIt = m_ioRegisters.find(0x1000E000u);
            const bool dmacEnabled = (dctrlIt == m_ioRegisters.end()) || ((dctrlIt->second & 0x1u) != 0u);
            if (!dmacEnabled)
            {
                return true;
            }

            const uint32_t channelBase = address & 0xFFFFFF00;
            const uint32_t madr = m_ioRegisters[channelBase + 0x10];
            const uint32_t qwc = m_ioRegisters[channelBase + 0x20];
            m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);

            if (tryProcessScratchpadDma(channelBase, value))
            {
                return true;
            }

            if ((channelBase == 0x1000A000u || channelBase == 0x10009000u || channelBase == 0x10008000u) && (m_gsVRAM || channelBase == 0x10008000u))
            {
                auto enqueueTransfer = [&](uint32_t srcAddr, uint32_t qwCount)
                {
                    if (qwCount == 0)
                        return;
                    const bool scratch = isScratchpad(srcAddr);
                    PendingTransfer pt;
                    pt.fromScratchpad = scratch;
                    pt.srcAddr = srcAddr;
                    pt.qwc = qwCount;
                    if (channelBase == 0x1000A000u)
                        m_pendingGifTransfers.push_back(pt);
                    else if (channelBase == 0x10009000u)
                        m_pendingVif1Transfers.push_back(pt);
                    else if (channelBase == 0x10008000u)
                        m_pendingVif0Transfers.push_back(pt);
                };

                uint32_t chcr = value;
                uint32_t mode = (chcr >> 2) & 0x3;

                // VIF1 with DIR=0 runs towards memory: a GS download (the data of a local->host
                // transfer comes out of the VIF1 FIFO). It used to be treated as an upload of
                // whatever was at MADR, so the game never saw the data - e.g. SSX 3 reads 16x8
                // Z-buffer blocks at each lens flare to test its visibility, and got stale RAM
                // (flares shone through terrain).
                if (channelBase == 0x10009000u && (value & 0x1u) == 0u && mode == 0u)
                {
                    const uint32_t bytes = (qwc & 0xFFFFu) * 16u;
                    uint32_t phys = 0u;
                    bool ok = bytes != 0u;
                    try
                    {
                        phys = translateAddress(madr);
                    }
                    catch (const std::exception &)
                    {
                        ok = false;
                    }
                    ok = ok && !isScratchpad(madr) && phys < PS2_RAM_SIZE && bytes <= PS2_RAM_SIZE - phys;
                    // A small probe the EE set up with the usual packet: answered from the last
                    // finished read at that place when there is queued work to wait for.
                    const ReadbackRequest req = s_rbRequest;
                    s_rbRequest.valid = false;
                    const bool probe = ok && req.valid && bytes >= req.pixelBytes && bytes <= 4u * kRbMaxPixels + 16u;
                    if (probe && gpuBusy() && rbServe(this, req, m_rdram + phys, bytes))
                    {
                        markModified(phys, bytes);
                    }
                    else
                    {
                        gpuSyncGlobal(kGpuSyncDma); // the GS commands that started the transfer run first
                        if (ok)
                        {
                            const uint32_t got = g_ps2GsReadbackHook ? g_ps2GsReadbackHook(m_rdram + phys, bytes) : 0u;
                            if (got < bytes)
                                std::memset(m_rdram + phys + got, 0, bytes - got);
                            markModified(phys, bytes);
                            if (probe && got >= req.pixelBytes)
                            {
                                rbStore(req, m_rdram + phys, bytes);
                                s_rbDirect.fetch_add(1u, std::memory_order_relaxed);
                            }
                        }
                    }
                    m_ioRegisters[channelBase + 0x10] = madr + bytes;
                    m_ioRegisters[channelBase + 0x20] = 0u;
                    m_ioRegisters[channelBase + 0x00] = value & ~0x100u;
                    static constexpr uint32_t kDStat = 0x1000E010u;
                    uint32_t dstat = m_ioRegisters.count(kDStat) ? m_ioRegisters[kDStat] : 0u;
                    dstat |= (1u << 1);
                    if (((dstat & 0x3FFu) & ((dstat >> 16) & 0x3FFu)) != 0u)
                        dstat |= (1u << 31);
                    else
                        dstat &= ~(1u << 31);
                    m_ioRegisters[kDStat] = dstat;
                    queueCompletedDmacCause(1u);
                    return true;
                }

                // Read-back sequence (see asyncReadbackEnabled): recognise its two PATH3 packets.
                RbPacket rbPacket = RbPacket::None;
                ReadbackRequest rbRequest;
                if (channelBase != 0x10008000u)
                    s_rbRequest.valid = false; // anything else sent to the GS ends a set-up download
                if (channelBase == 0x1000A000u && mode == 0 && (qwc == 2u || qwc == 5u) && asyncReadbackEnabled())
                {
                    try
                    {
                        const uint32_t src = translateAddress(madr);
                        const uint8_t *base = isScratchpad(madr) ? m_scratchpad : m_rdram;
                        const uint32_t limit = isScratchpad(madr) ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
                        if (src < limit && qwc * 16u <= limit - src)
                            rbPacket = rbClassify(base + src, qwc, rbRequest);
                    }
                    catch (const std::exception &)
                    {
                    }
                }

                if (mode == 0 && qwc > 0)
                {
                    enqueueTransfer(madr, qwc);
                }
                else if (mode == 1)
                {
                    uint32_t tagAddr = m_ioRegisters[channelBase + 0x30];
                    uint32_t asr0 = m_ioRegisters[channelBase + 0x40];
                    uint32_t asr1 = m_ioRegisters[channelBase + 0x50];
                    uint32_t asp = (chcr >> 4) & 0x3u;
                    const bool tieEnabled = (chcr & (1u << 7)) != 0u;
                    // Safety cap only (runaway/cyclic chains). SSX 3's per-frame display list is
                    // well over 4096 tags in busy scenes; the old 4096 cap silently dropped the rest
                    // of the frame (riders, hair and hats, which are drawn late).
                    const int kMaxChainTags = 1 << 20;
                    std::vector<uint8_t> chainBuf = gpuTakeBuffer();

                    auto appendData = [&](uint32_t srcAddr, uint32_t qwCount)
                    {
                        const uint64_t bytes64 = static_cast<uint64_t>(qwCount) * 16ull;
                        uint32_t bytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
                        const bool scratch = isScratchpad(srcAddr);
                        uint32_t src = 0;
                        src = translateAddress(srcAddr);
                        const uint8_t *base2;
                        uint32_t maxSz2;
                        if (scratch)
                        {
                            base2 = m_scratchpad;
                            maxSz2 = PS2_SCRATCHPAD_SIZE;
                        }
                        else
                        {
                            base2 = m_rdram;
                            maxSz2 = PS2_RAM_SIZE;
                        }

                        while (bytes > 0)
                        {
                            if (src >= maxSz2)
                                src = 0;
                            uint32_t chunk = bytes;
                            if (src + chunk > maxSz2)
                                chunk = maxSz2 - src;
                            if (chunk == 0)
                                break;
                            chainBuf.insert(chainBuf.end(), base2 + src, base2 + src + chunk);
                            bytes -= chunk;
                            src += chunk;
                        }
                    };

                    auto appendVifTagData = [&](uint32_t localTagAddr)
                    {
                        uint32_t tagPhys = 0u;
                        const bool tagScratch = isScratchpad(localTagAddr);
                        tagPhys = translateAddress(localTagAddr);

                        const uint8_t *localBase = tagScratch ? m_scratchpad : m_rdram;
                        const uint32_t localMax = tagScratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
                        if (tagPhys + 16u > localMax)
                            return;

                        // CHCR.TTE sends the DMAtag's upper 64 bits to the channel before
                        // the tag payload. VIF chains use those bytes for two VIFcodes.
                        chainBuf.insert(chainBuf.end(), localBase + tagPhys + 8u, localBase + tagPhys + 16u);
                    };

                    const bool isVifChannel =
                        channelBase == 0x10009000u || channelBase == 0x10008000u;
                    const bool transferTagData = isVifChannel && (chcr & 0x40u) != 0u;

                    int tagsProcessed = 0;
                    uint32_t lastTagUpper = (chcr >> 16) & 0xFFFFu;

                    while (tagsProcessed < kMaxChainTags)
                    {
                        const uint32_t currentTagAddr = tagAddr;
                        const bool tagInSPR = isScratchpad(tagAddr);
                        uint32_t physTag = 0;
                        try
                        {
                            physTag = translateAddress(tagAddr);
                        }
                        catch (...)
                        {
                            break;
                        }
                        const uint8_t *tagBase;
                        uint32_t tagMax;
                        if (tagInSPR)
                        {
                            tagBase = m_scratchpad;
                            tagMax = PS2_SCRATCHPAD_SIZE;
                        }
                        else
                        {
                            tagBase = m_rdram;
                            tagMax = PS2_RAM_SIZE;
                        }
                        if (physTag + 16 > tagMax)
                            break;

                        const uint8_t *tp = tagBase + physTag;
                        uint64_t tag = loadScalar<uint64_t>(tp, 0, 16, "dma chain tag", tagAddr);
                        uint16_t tagQwc = static_cast<uint16_t>(tag & 0xFFFF);
                        uint32_t id = static_cast<uint32_t>((tag >> 28) & 0x7);
                        const bool irq = ((tag >> 31) & 0x1ull) != 0ull;
                        uint32_t addr = static_cast<uint32_t>((tag >> 32) & 0x7FFFFFFF);
                        lastTagUpper = static_cast<uint32_t>((tag >> 16) & 0xFFFFu);
                        ++tagsProcessed;

                        uint32_t dataAddr = 0;
                        bool hasPayload = (tagQwc > 0);
                        bool endChain = false;

                        switch (id)
                        {
                        case 0:
                            dataAddr = addr;
                            tagAddr = tagAddr + 16;
                            endChain = true;
                            break;
                        case 1:
                            dataAddr = tagAddr + 16;
                            tagAddr = dataAddr + static_cast<uint32_t>(tagQwc) * 16u;
                            break;
                        case 2:
                            dataAddr = tagAddr + 16;
                            tagAddr = addr;
                            break;
                        case 3:
                        case 4:
                            dataAddr = addr;
                            tagAddr = tagAddr + 16;
                            break;
                        case 5:
                            dataAddr = tagAddr + 16;
                            {
                                const uint32_t retAddr = dataAddr + static_cast<uint32_t>(tagQwc) * 16u;
                                if (asp == 0u)
                                {
                                    asr0 = retAddr;
                                    asp = 1u;
                                }
                                else if (asp == 1u)
                                {
                                    asr1 = retAddr;
                                    asp = 2u;
                                }
                            }
                            tagAddr = addr;
                            break;
                        case 6:
                            dataAddr = tagAddr + 16;
                            if (asp == 2u)
                            {
                                tagAddr = asr1;
                                asp = 1u;
                            }
                            else if (asp == 1u)
                            {
                                tagAddr = asr0;
                                asp = 0u;
                            }
                            else
                            {
                                endChain = true;
                            }
                            break;
                        case 7:
                            dataAddr = tagAddr + 16;
                            endChain = true;
                            break;
                        default:
                            hasPayload = false;
                            endChain = true;
                            break;
                        }

                        if (transferTagData)
                            appendVifTagData(currentTagAddr);

                        if (hasPayload)
                            appendData(dataAddr, tagQwc);
                        if (irq && tieEnabled)
                            endChain = true;
                        if (endChain)
                            break;
                    }

                    if (tagsProcessed >= kMaxChainTags)
                    {
                        static uint32_t capLogs = 0u;
                        if (capLogs++ < 8u)
                            std::fprintf(stderr, "[dma] chain on 0x%x hit the %d-tag safety cap at tag 0x%x\n",
                                         (unsigned)channelBase, kMaxChainTags, (unsigned)tagAddr);
                    }
                    m_ioRegisters[channelBase + 0x30] = tagAddr;
                    m_ioRegisters[channelBase + 0x40] = asr0;
                    m_ioRegisters[channelBase + 0x50] = asr1;
                    chcr = (chcr & ~(0x3u << 4)) | ((asp & 0x3u) << 4);
                    chcr = (chcr & 0x0000FFFFu) | (lastTagUpper << 16);
                    m_ioRegisters[channelBase + 0x00] = chcr;

                    if (!chainBuf.empty())
                    {
                        PendingTransfer pt;
                        pt.fromScratchpad = false;
                        pt.srcAddr = 0;
                        pt.qwc = 0;
                        pt.chainData = std::move(chainBuf);
                        if (channelBase == 0x1000A000)
                        {
                            m_pendingGifTransfers.push_back(std::move(pt));
                        }
                        else if (channelBase == 0x10009000u)
                        {
                            m_pendingVif1Transfers.push_back(std::move(pt));
                        }
                        else if (channelBase == 0x10008000u)
                        {
                            m_pendingVif0Transfers.push_back(std::move(pt));
                        }
                    }
                    // else if (channelBase == 0x10009000u)
                    // {

                    // }
                }
                else if (qwc > 0)
                {
                    enqueueTransfer(madr, qwc);
                }

                const bool autoProcessTransfers =
                    (channelBase == 0x1000A000u) ? (m_gifPacketCallback || m_gifArbiter != nullptr) : true;
                if (autoProcessTransfers)
                {
                    // VIF1 transfers towards memory (GS downloads) stay synchronous.
                    const bool toMemory = channelBase == 0x10009000u && (value & 0x1u) == 0u;
                    processPendingTransfers(!toMemory);
                    if (rbPacket == RbPacket::Finish && gpuBusy())
                    {
                        // FINISH as soon as the packet is queued (the queued one is dropped if
                        // the EE clears FINISH before the GS thread gets to it).
                        std::lock_guard<std::mutex> lock(s_finishMutex);
                        gs_regs.csr.fetch_or(0x2u);
                        s_finishEarly.fetch_add(1u, std::memory_order_relaxed);
                    }
                    else if (rbPacket == RbPacket::Download)
                        s_rbRequest = rbRequest;
                }
            }
        }
        return true;
    }

    if (address >= 0x10000000 && address < 0x10010000)
    {
        if (address >= 0x10000200 && address < 0x10000300)
        {
            return true;
        }
        if (address >= 0x10000000 && address < 0x10000100)
        {
            return true;
        }
    }

    return false;
}

bool PS2Memory::tryProcessScratchpadDma(uint32_t channelBase, uint32_t chcr)
{
    static constexpr uint32_t kSprFromChannel = 0x1000D000u;
    static constexpr uint32_t kSprToChannel = 0x1000D400u;
    if (channelBase != kSprFromChannel && channelBase != kSprToChannel)
        return false;

    const uint32_t mode = (chcr >> 2u) & 0x3u;
    if (mode != 0u)
        return false;

    const uint32_t qwc = m_ioRegisters[channelBase + 0x20u] & 0xFFFFu;
    const uint32_t byteCount = qwc * 16u;
    const uint32_t originalMadr = m_ioRegisters[channelBase + 0x10u] & 0x7FFFFFF0u;
    const uint32_t originalSadr = m_ioRegisters[channelBase + 0x80u] & 0x3FF0u;

    uint32_t mainOffset = 0u;
    try
    {
        mainOffset = translateAddress(originalMadr);
    }
    catch (const std::exception &)
    {
        return false;
    }

    if (mainOffset > PS2_RAM_SIZE || byteCount > PS2_RAM_SIZE - mainOffset)
        return false;

    const bool fromScratchpad = channelBase == kSprFromChannel;
    uint32_t scratchOffset = originalSadr;
    uint32_t bytesLeft = byteCount;
    uint32_t copied = 0u;
    while (bytesLeft != 0u)
    {
        const uint32_t scratchChunk = PS2_SCRATCHPAD_SIZE - scratchOffset;
        const uint32_t chunk = std::min(bytesLeft, scratchChunk);
        if (fromScratchpad)
        {
            std::memcpy(m_rdram + mainOffset + copied, m_scratchpad + scratchOffset, chunk);
            markModified(mainOffset + copied, chunk);
        }
        else
        {
            std::memcpy(m_scratchpad + scratchOffset, m_rdram + mainOffset + copied, chunk);
        }

        copied += chunk;
        bytesLeft -= chunk;
        scratchOffset = (scratchOffset + chunk) & (PS2_SCRATCHPAD_SIZE - 1u);
    }

    m_ioRegisters[channelBase + 0x10u] = (originalMadr + byteCount) & 0x7FFFFFF0u;
    m_ioRegisters[channelBase + 0x20u] = 0u;
    m_ioRegisters[channelBase + 0x80u] = (originalSadr + byteCount) & 0x3FF0u;
    completeDmacChannel(channelBase, fromScratchpad ? 8u : 9u);
    return true;
}

void PS2Memory::completeDmacChannel(uint32_t channelBase, uint32_t cause)
{
    static constexpr uint32_t kDStat = 0x1000E010u;
    m_ioRegisters[channelBase] &= ~0x100u;

    uint32_t dstat = m_ioRegisters.count(kDStat) ? m_ioRegisters[kDStat] : 0u;
    dstat |= 1u << cause;
    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask = (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
        dstat |= 1u << 31u;
    else
        dstat &= ~(1u << 31u);
    m_ioRegisters[kDStat] = dstat;
    queueCompletedDmacCause(cause);
}

void PS2Memory::processPendingTransfers(bool allowAsync)
{
    const bool hadGif = !m_pendingGifTransfers.empty();
    uint32_t observedGifQwc = 0u;
    for (const auto &transfer : m_pendingGifTransfers)
    {
        const uint64_t transferQwc = !transfer.chainData.empty()
                                         ? (transfer.chainData.size() / 16u)
                                         : transfer.qwc;
        observedGifQwc = static_cast<uint32_t>(std::min<uint64_t>(16u, static_cast<uint64_t>(observedGifQwc) + transferQwc));
    }
    if (observedGifQwc != 0u)
    {
        constexpr uint32_t kGifStat = 0x10003020u;
        constexpr uint32_t kGifFqcMask = 0x1F000000u;
        uint32_t &gifStat = m_ioRegisters[kGifStat];
        gifStat = (gifStat & ~kGifFqcMask) | (observedGifQwc << 24u);
    }

    // Capture every GIF / VIF1 transfer into an owned buffer: the EE may overwrite the source as
    // soon as the channel reports completion (which happens right below).
    auto materialize = [&](PendingTransfer &p, std::vector<uint8_t> &out) -> bool
    {
        if (!p.chainData.empty())
        {
            out = std::move(p.chainData);
            return true;
        }
        if (p.qwc == 0u)
            return false;
        uint32_t srcPhys = 0;
        try
        {
            srcPhys = translateAddress(p.srcAddr);
        }
        catch (const std::exception &)
        {
            return false;
        }
        const uint8_t *base = p.fromScratchpad ? m_scratchpad : m_rdram;
        const uint32_t limit = p.fromScratchpad ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
        const uint64_t bytes64 = std::min<uint64_t>(static_cast<uint64_t>(p.qwc) * 16ull, limit);
        uint32_t bytesLeft = static_cast<uint32_t>(bytes64);
        out.resize(bytesLeft);
        uint32_t written = 0u;
        while (bytesLeft > 0u)
        {
            if (srcPhys >= limit)
                srcPhys = 0u;
            const uint32_t chunk = std::min(bytesLeft, limit - srcPhys);
            std::memcpy(out.data() + written, base + srcPhys, chunk);
            written += chunk;
            bytesLeft -= chunk;
            srcPhys += chunk;
        }
        return true;
    };

    GpuJob job;
    job.countKick = true;
    for (auto &p : m_pendingGifTransfers)
    {
        std::vector<uint8_t> data;
        if (materialize(p, data) && data.size() >= 16u)
        {
            m_seenGifCopy = true;
            m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
            job.gif.push_back(std::move(data));
        }
    }
    m_pendingGifTransfers.clear();

    const bool hadVif0 = !m_pendingVif0Transfers.empty();
    for (auto &p : m_pendingVif0Transfers)
    {
        if (!p.chainData.empty())
        {
            processVIF0Data(p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
        }
        else if (p.qwc > 0)
        {
            uint32_t srcPhys = 0;
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF0Data(m_scratchpad + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF0Data(srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingVif0Transfers.clear();

    const bool hadVif1 = !m_pendingVif1Transfers.empty();
    for (auto &p : m_pendingVif1Transfers)
    {
        std::vector<uint8_t> data;
        if (materialize(p, data) && !data.empty())
            job.vif1.push_back(std::move(data));
    }
    m_pendingVif1Transfers.clear();

    if (allowAsync && gpuAsyncEnabled())
    {
        if (!job.gif.empty() || !job.vif1.empty())
            gpuEnqueue(this, std::move(job));
    }
    else
    {
        gpuSyncGlobal(kGpuSyncDma);
        runGpuJob(job);
    }

    static constexpr uint32_t GIF_CHANNEL = 0x1000A000;
    static constexpr uint32_t VIF0_CHANNEL = 0x10008000;
    static constexpr uint32_t VIF1_CHANNEL = 0x10009000;
    static constexpr uint32_t D_STAT = 0x1000E010u;

    auto raiseDStatChannel = [&](uint32_t channelBit)
    {
        uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
        dstat |= (1u << channelBit);

        const uint32_t status = dstat & 0x3FFu;
        const uint32_t mask = (dstat >> 16) & 0x3FFu;
        if ((status & mask) != 0u)
            dstat |= (1u << 31);
        else
            dstat &= ~(1u << 31);

        m_ioRegisters[D_STAT] = dstat;
    };

    if (hadGif)
    {
        raiseDStatChannel(2u); // GIF channel
        queueCompletedDmacCause(2u);
        m_ioRegisters[GIF_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[GIF_CHANNEL + 0x20] = 0;
    }
    if (hadVif0)
    {
        raiseDStatChannel(0u); // VIF0 channel
        queueCompletedDmacCause(0u);
        m_ioRegisters[VIF0_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[VIF0_CHANNEL + 0x20] = 0;
    }
    if (hadVif1)
    {
        raiseDStatChannel(1u); // VIF1 channel
        queueCompletedDmacCause(1u);
        m_ioRegisters[VIF1_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[VIF1_CHANNEL + 0x20] = 0;
    }
}

void PS2Memory::queueCompletedDmacCause(uint32_t cause)
{
    std::lock_guard<std::mutex> lock(m_completedDmacMutex);
    m_completedDmacCauses.push_back(cause);
}

std::vector<uint32_t> PS2Memory::consumeCompletedDmacCauses()
{
    std::lock_guard<std::mutex> lock(m_completedDmacMutex);
    std::vector<uint32_t> causes;
    causes.swap(m_completedDmacCauses);
    return causes;
}

uint32_t g_path3ReleaseLimit = 0u; // see flushMaskedPath3Packets / VIF MSKPATH3

void PS2Memory::flushMaskedPath3Packets(bool drainImmediately)
{
    if (!t_onGpuWorker && gpuBusy())
        gpuSyncGlobal(kGpuSyncOther);
    if (m_path3Masked || m_path3MaskedFifo.empty())
        return;

    auto emit = [&](const uint8_t *packetData, uint32_t packetSize)
    {
        if (m_gifArbiter)
            m_gifArbiter->submit(GifPathId::Path3, packetData, packetSize, false);
        else if (m_gifPacketCallback)
            m_gifPacketCallback(packetData, packetSize);
    };

    // g_path3ReleaseLimit (set by the VIF MSKPATH3 handler) limits how many queued PATH3 packets
    // an unmask window releases; 0 = all.
    const size_t limit = g_path3ReleaseLimit != 0u ? std::min<size_t>(g_path3ReleaseLimit, m_path3MaskedFifo.size())
                                                   : m_path3MaskedFifo.size();
    for (size_t i = 0; i < limit; ++i)
    {
        const auto &packet = m_path3MaskedFifo[i];
        if (packet.size() >= 16u)
            emit(packet.data(), static_cast<uint32_t>(packet.size()));
    }
    m_path3MaskedFifo.erase(m_path3MaskedFifo.begin(), m_path3MaskedFifo.begin() + static_cast<std::ptrdiff_t>(limit));

    if (m_gifArbiter && drainImmediately)
        m_gifArbiter->drain();
}

void PS2Memory::submitGifPacket(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool drainImmediately, bool path2DirectHl)
{
    if (!t_onGpuWorker && gpuBusy())
        gpuSyncGlobal(kGpuSyncOther); // GIF state belongs to the GPU worker while it runs
    if (!data || sizeBytes < 16)
        return;

    if (pathId == GifPathId::Path3)
        ssx3FrameRecord(*this, 'G', nullptr, 0u, data, sizeBytes);

    if (pathId == GifPathId::Path3)
    {
        if (m_path3Masked)
        {
            // PATH3 masking works at GIF packet granularity (a packet ends at a tag with EOP=1):
            // each MSKPATH3 unmask window lets the next whole packet through. Queue a DMA chain as
            // its individual EOP-terminated packets so the windows can release them one at a time.
            uint32_t packetStart = 0u;
            uint32_t pos = 0u;
            while (pos + 16u <= sizeBytes)
            {
                uint64_t tagLo = 0;
                uint64_t tagHi = 0;
                std::memcpy(&tagLo, data + pos, 8);
                std::memcpy(&tagHi, data + pos + 8, 8);
                pos += 16u;
                const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
                const bool eop = ((tagLo >> 15) & 1u) != 0u;
                const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 3u);
                uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
                if (nreg == 0u)
                    nreg = 16u;
                uint64_t payload = 0u;
                if (flg == 0u)
                    payload = static_cast<uint64_t>(nloop) * nreg * 16u;
                else if (flg == 1u)
                    payload = ((static_cast<uint64_t>(nloop) * nreg * 8u) + 15u) & ~static_cast<uint64_t>(15u);
                else
                    payload = static_cast<uint64_t>(nloop) * 16u;
                pos = static_cast<uint32_t>(std::min<uint64_t>(sizeBytes, pos + payload));
                if (eop || pos >= sizeBytes)
                {
                    m_path3MaskedFifo.emplace_back(data + packetStart, data + pos);
                    packetStart = pos;
                }
            }
            if (packetStart < sizeBytes && sizeBytes - packetStart >= 16u)
                m_path3MaskedFifo.emplace_back(data + packetStart, data + sizeBytes);
            return;
        }
        flushMaskedPath3Packets(false);
    }

    if (m_gifArbiter)
        m_gifArbiter->submit(pathId, data, sizeBytes, path2DirectHl);
    else if (m_gifPacketCallback)
        m_gifPacketCallback(data, sizeBytes);

    if (m_gifArbiter && drainImmediately)
        m_gifArbiter->drain();
}

void PS2Memory::processGIFPacket(uint32_t srcPhysAddr, uint32_t qwCount)
{
    if (!m_rdram || qwCount == 0)
        return;
    const uint64_t bytes64 = static_cast<uint64_t>(qwCount) * 16ull;
    uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
    uint32_t bytesLeft = sizeBytes;
    while (bytesLeft >= 16)
    {
        if (srcPhysAddr >= PS2_RAM_SIZE)
            srcPhysAddr = 0;
        uint32_t chunk = bytesLeft;
        if (srcPhysAddr + chunk > PS2_RAM_SIZE)
            chunk = PS2_RAM_SIZE - srcPhysAddr;
        if (chunk == 0)
            break;

        m_seenGifCopy = true;
        m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
        submitGifPacket(GifPathId::Path3, m_rdram + srcPhysAddr, chunk);

        bytesLeft -= chunk;
        srcPhysAddr += chunk;
    }
}

void PS2Memory::processGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (m_gifArbiter)
        submitGifPacket(GifPathId::Path3, data, sizeBytes);
    else if (m_gifPacketCallback && data && sizeBytes >= 16)
        m_gifPacketCallback(data, sizeBytes);
}

bool PS2Memory::tryProcessNativeGifImageUploadChain(GS &gs, uint32_t tadr, uint32_t chcr)
{
    static constexpr uint32_t GIF_CHANNEL = 0x1000A000u;
    static constexpr uint32_t D_STAT = 0x1000E010u;
    static constexpr uint32_t D_CTRL = 0x1000E000u;

    if (!m_rdram || !m_gsVRAM || m_path3Masked)
        return false;
    if (m_gifArbiter && !m_gifArbiter->empty())
        return false;
    if ((chcr & 0x100u) == 0u || ((chcr >> 2u) & 0x3u) != 1u)
        return false;
    if ((chcr & (1u << 7u)) != 0u || ((chcr >> 4u) & 0x3u) != 0u)
        return false;

    const auto dctrlIt = m_ioRegisters.find(D_CTRL);
    if (dctrlIt != m_ioRegisters.end() && ((dctrlIt->second & 0x1u) == 0u))
        return false;

    auto resolveContiguous = [&](uint32_t guestAddr, uint32_t bytes, const uint8_t *&out) -> bool
    {
        try
        {
            const bool scratch = isScratchpad(guestAddr);
            const uint32_t phys = translateAddress(guestAddr);
            const uint8_t *base = scratch ? m_scratchpad : m_rdram;
            const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
            if (!base || phys > limit || bytes > limit - phys)
                return false;
            out = base + phys;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    };

    auto loadDmaTagAt = [&](uint32_t guestAddr, DmaTagView &out) -> bool
    {
        const uint8_t *ptr = nullptr;
        if (!resolveContiguous(guestAddr, 16u, ptr))
            return false;
        out = decodeDmaTag(loadScalar<uint64_t>(ptr, 0u, 16u, "native gif dma tag", guestAddr));
        return true;
    };

    auto decodeSetupPayload = [&](const uint8_t *payload, uint64_t (&regs)[4]) -> bool
    {
        const uint64_t tagLo = loadScalar<uint64_t>(payload, 0u, 80u, "native gif setup tag", 0u);
        const uint64_t tagHi = loadScalar<uint64_t>(payload, 8u, 80u, "native gif setup regs", 0u);
        if (gifTagNloop(tagLo) != 4u ||
            gifTagFlg(tagLo) != GIF_FMT_PACKED ||
            gifTagNreg(tagLo) != 1u ||
            (tagHi & 0xFull) != 0x0Eull)
        {
            return false;
        }

        static constexpr uint8_t kExpectedRegs[4] = {
            GS_REG_BITBLTBUF,
            GS_REG_TRXPOS,
            GS_REG_TRXREG,
            GS_REG_TRXDIR,
        };

        uint32_t offset = 16u;
        for (uint32_t i = 0; i < 4u; ++i)
        {
            regs[i] = loadScalar<uint64_t>(payload, offset, 80u, "native gif setup value", 0u);
            const uint64_t reg = loadScalar<uint64_t>(payload, offset + 8u, 80u, "native gif setup register", 0u);
            if ((reg & 0xFFu) != kExpectedRegs[i])
                return false;
            offset += 16u;
        }

        const uint32_t trxdirMode = static_cast<uint32_t>(regs[3] & 0x3ull);
        const uint32_t rrw = static_cast<uint32_t>(regs[2] & 0xFFFull);
        const uint32_t rrh = static_cast<uint32_t>((regs[2] >> 32u) & 0xFFFull);
        return trxdirMode == 0u && rrw != 0u && rrh != 0u;
    };

    DmaTagView setupTag{};
    if (!loadDmaTagAt(tadr, setupTag) ||
        setupTag.id != 1u ||
        setupTag.qwc != 5u ||
        setupTag.irq)
    {
        return false;
    }

    const uint8_t *setupPayload = nullptr;
    const uint32_t setupPayloadAddr = tadr + 16u;
    if (!resolveContiguous(setupPayloadAddr, 5u * 16u, setupPayload))
        return false;

    uint64_t setupRegs[4] = {};
    if (!decodeSetupPayload(setupPayload, setupRegs))
        return false;

    uint32_t imageTagDmaAddr = setupPayloadAddr + 5u * 16u;
    DmaTagView imageTagDma{};
    if (!loadDmaTagAt(imageTagDmaAddr, imageTagDma) ||
        imageTagDma.id != 1u ||
        imageTagDma.qwc != 1u ||
        imageTagDma.irq)
    {
        return false;
    }

    const uint8_t *imageGifTag = nullptr;
    if (!resolveContiguous(imageTagDmaAddr + 16u, 16u, imageGifTag))
        return false;

    const uint64_t imageTagLo = loadScalar<uint64_t>(imageGifTag, 0u, 16u, "native gif image tag", imageTagDmaAddr + 16u);
    if (gifTagFlg(imageTagLo) != GIF_FMT_IMAGE)
        return false;

    const uint32_t imageQwc = gifTagNloop(imageTagLo);
    if (imageQwc == 0u)
        return false;

    const uint64_t imageBytes64 = static_cast<uint64_t>(imageQwc) * 16ull;
    if (imageBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t imageBytes = static_cast<uint32_t>(imageBytes64);

    const uint32_t payloadTagAddr = imageTagDmaAddr + 32u;
    DmaTagView payloadTag{};
    if (!loadDmaTagAt(payloadTagAddr, payloadTag) ||
        payloadTag.qwc != imageQwc ||
        payloadTag.irq)
    {
        return false;
    }

    uint32_t imageDataAddr = 0u;
    uint32_t finalTadr = payloadTagAddr;
    uint32_t lastTagUpper = payloadTag.upper;
    if (payloadTag.id == 3u || payloadTag.id == 4u)
    {
        imageDataAddr = payloadTag.addr;
        const uint32_t terminalTagAddr = payloadTagAddr + 16u;
        DmaTagView terminalTag{};
        if (!loadDmaTagAt(terminalTagAddr, terminalTag) ||
            terminalTag.qwc != 0u ||
            terminalTag.irq ||
            (terminalTag.id != 0u && terminalTag.id != 7u))
        {
            return false;
        }
        finalTadr = (terminalTag.id == 0u) ? (terminalTagAddr + 16u) : terminalTagAddr;
        lastTagUpper = terminalTag.upper;
    }
    else if (payloadTag.id == 7u)
    {
        imageDataAddr = payloadTagAddr + 16u;
        finalTadr = payloadTagAddr;
    }
    else
    {
        return false;
    }

    const uint8_t *imageData = nullptr;
    if (!resolveContiguous(imageDataAddr, imageBytes, imageData))
        return false;

    m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);
    m_seenGifCopy = true;
    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
    gs.uploadImageNative(setupRegs[0], setupRegs[1], setupRegs[2], setupRegs[3], imageData, imageBytes);

    m_ioRegisters[GIF_CHANNEL + 0x30u] = finalTadr;
    m_ioRegisters[GIF_CHANNEL + 0x40u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x50u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x00u] = ((chcr & 0x0000FFFFu) | (lastTagUpper << 16u)) & ~0x100u;
    m_ioRegisters[GIF_CHANNEL + 0x20u] = 0u;

    uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
    dstat |= (1u << 2u);
    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask = (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
        dstat |= (1u << 31u);
    else
        dstat &= ~(1u << 31u);
    m_ioRegisters[D_STAT] = dstat;
    queueCompletedDmacCause(2u);
    return true;
}

bool PS2Memory::tryProcessNativeGifPackedChain(GS &gs, uint32_t tadr, uint32_t chcr)
{
    static constexpr uint32_t GIF_CHANNEL = 0x1000A000u;
    static constexpr uint32_t D_STAT = 0x1000E010u;
    static constexpr uint32_t D_CTRL = 0x1000E000u;

    if (!m_rdram || !m_gsVRAM || m_path3Masked)
        return false;
    if (m_gifArbiter && !m_gifArbiter->empty())
        return false;
    if ((chcr & 0x100u) == 0u || ((chcr >> 2u) & 0x3u) != 1u)
        return false;
    if ((chcr & (1u << 7u)) != 0u || ((chcr >> 4u) & 0x3u) != 0u)
        return false;

    const auto dctrlIt = m_ioRegisters.find(D_CTRL);
    if (dctrlIt != m_ioRegisters.end() && ((dctrlIt->second & 0x1u) == 0u))
        return false;

    auto resolveContiguous = [&](uint32_t guestAddr, uint32_t bytes, const uint8_t *&out) -> bool
    {
        try
        {
            const bool scratch = isScratchpad(guestAddr);
            const uint32_t phys = translateAddress(guestAddr);
            const uint8_t *base = scratch ? m_scratchpad : m_rdram;
            const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
            if (!base || phys > limit || bytes > limit - phys)
                return false;
            out = base + phys;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    };

    const uint8_t *tagPtr = nullptr;
    if (!resolveContiguous(tadr, 16u, tagPtr))
        return false;

    const DmaTagView tag = decodeDmaTag(loadScalar<uint64_t>(tagPtr, 0u, 16u, "native packed gif dma tag", tadr));
    if (tag.id != 7u || tag.qwc == 0u || tag.irq)
        return false;

    const uint64_t payloadBytes64 = static_cast<uint64_t>(tag.qwc) * 16ull;
    if (payloadBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t payloadBytes = static_cast<uint32_t>(payloadBytes64);

    const uint8_t *payload = nullptr;
    if (!resolveContiguous(tadr + 16u, payloadBytes, payload))
        return false;
    if (!gs.processNativePackedGIFPacket(payload, payloadBytes))
        return false;

    m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);
    m_seenGifCopy = true;
    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);

    m_ioRegisters[GIF_CHANNEL + 0x30u] = tadr;
    m_ioRegisters[GIF_CHANNEL + 0x40u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x50u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x00u] = ((chcr & 0x0000FFFFu) | (tag.upper << 16u)) & ~0x100u;
    m_ioRegisters[GIF_CHANNEL + 0x20u] = 0u;

    uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
    dstat |= (1u << 2u);
    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask = (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
        dstat |= (1u << 31u);
    else
        dstat &= ~(1u << 31u);
    m_ioRegisters[D_STAT] = dstat;
    queueCompletedDmacCause(2u);
    return true;
}

int PS2Memory::pollDmaRegisters()
{
    return 0;
}

uint32_t PS2Memory::readIORegister(uint32_t address)
{
    size_t timerIndex = 0u;
    uint32_t timerOffset = 0u;
    if (decodeEeTimerRegister(address, timerIndex, timerOffset))
    {
        const EeTimer &timer = m_eeTimers[timerIndex];
        switch (timerOffset)
        {
        case kEeTimerCountOffset:
            return timer.count & 0xFFFFu;
        case kEeTimerModeOffset:
            return timer.mode & (kEeTimerModeConfigMask | kEeTimerModeStatusMask);
        case kEeTimerCompareOffset:
            return timer.compare & 0xFFFFu;
        case kEeTimerHoldOffset:
            return timer.hold & 0xFFFFu;
        default:
            return 0u;
        }
    }

    if (isGsPrivReg(address))
    {
        // NB: unreachable from read8/16/32/64 today, same reasoning as the write
        // path above; kept correct for direct callers.
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            return static_cast<uint32_t>((gsCsrReadValue(gs_regs.csr.load()) >> (off * 8u)) & 0xFFFFFFFFull);
        }
        if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            return static_cast<uint32_t>((*reg >> (off * 8u)) & 0xFFFFFFFFull);
        }
        return 0u;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        uint32_t val = 0;
        switch (address)
        {
        case 0x10002000:
            val = m_ioRegisters[address];
            break;
        case 0x10002010:
            val = m_ioRegisters[address] & ~(1u << 31);
            break;
        case 0x10002020:
        case 0x10002030:
            val = m_ioRegisters[address];
            break;
        default:
            val = 0;
            break;
        }
        return val;
    }

    if (address == 0x10003020u) // GIF_STAT
    {
        uint32_t stat = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        const uint32_t mode = m_ioRegisters.count(0x10003010u) ? m_ioRegisters[0x10003010u] : 0u;
        const uint32_t ctrl = m_ioRegisters.count(0x10003000u) ? m_ioRegisters[0x10003000u] : 0u;

        // M3R and IMT mirror GIF_MODE, PSE mirrors GIF_CTRL, and M3P is the
        // effective PATH3 mask controlled by the VIF1 MSKPATH3 command.
        stat = (stat & ~0xFu) |
               (mode & 0x1u) |
               (m_path3Masked ? 0x2u : 0u) |
               (mode & 0x4u) |
               (ctrl & 0x8u);
        return stat;
    }

    if (address >= 0x10000000 && address < 0x10010000)
    {
        if (address >= 0x10008000 && address < 0x1000F000)
        {
            if ((address & 0xFF) == 0x00)
            {
                uint32_t channelStatus = m_ioRegisters[address] & ~0x100u;
                m_ioRegisters[address] = channelStatus;
                return channelStatus;
            }
        }

        if (address >= 0x10000200 && address < 0x10000300)
        {
            return 0;
        }

        if (address >= 0x1000F200 && address <= 0x1000F260)
        {
            if (address == 0x1000F230)
            {
                return 0x60000;
            }
            if (address == 0x1000F240)
            {
                return 0xF0000002;
            }
            return 0;
        }
    }

    auto it = m_ioRegisters.find(address);
    if (it != m_ioRegisters.end())
    {
        return it->second;
    }

    return 0;
}

void PS2Memory::registerCodeRegion(uint32_t start, uint32_t end)
{
    if (end <= start)
    {
        std::cerr << "Ignoring invalid code region: start=0x" << std::hex << start
                  << " end=0x" << end << std::dec << std::endl;
        return;
    }

    if ((end - start) > PS2_RAM_SIZE)
    {
        std::cerr << "Ignoring oversized code region: start=0x" << std::hex << start
                  << " end=0x" << end << std::dec << std::endl;
        return;
    }

    for (const auto &existing : m_codeRegions)
    {
        if (existing.start == start && existing.end == end)
        {
            return;
        }
    }

    CodeRegion region;
    region.start = start;
    region.end = end;

    size_t sizeInWords = (end - start + 3u) / 4u;
    region.modified.resize(sizeInWords, false);

    m_codeRegions.push_back(region);
    RUNTIME_LOG("Registered code region: " << std::hex << start << " - " << end << std::dec);
}

bool PS2Memory::isAddressInRegion(uint32_t address, const CodeRegion &region)
{
    return (address >= region.start && address < region.end);
}

bool PS2Memory::isCodeAddress(uint32_t address) const
{
    for (const auto &region : m_codeRegions)
    {
        if (address >= region.start && address < region.end)
        {
            return true;
        }
    }
    return false;
}

void PS2Memory::markModified(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size())
            {
                region.modified[bitIndex] = true;
                RUNTIME_LOG("Marked code at " << std::hex << addr << std::dec << " as modified");
            }
        }
    }
}

bool PS2Memory::isCodeModified(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return false;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (const auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size() && region.modified[bitIndex])
            {
                return true; // Found modified code
            }
        }
    }

    return false; // No modifications found
}

void PS2Memory::clearModifiedFlag(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size())
            {
                region.modified[bitIndex] = false;
            }
        }
    }
}
