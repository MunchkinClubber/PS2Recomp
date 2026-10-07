#include "runtime/gs/gs_interp_backend.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "runtime/ps2_hitch.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

extern std::atomic<uint64_t> g_ssx3GsKicks; // gs_frontend.cpp: vertex kicks so far (GS thread)

extern std::atomic<uint64_t> g_ps2FlipIssuedNs; // ps2_memory.cpp: when the EE wrote the flip being processed

void (*g_gsInterpPassHook)(GSRasterBackend *inner, uint32_t pass, uint32_t passes, float t) = nullptr;

namespace
{
    constexpr uint32_t kNone = 0xFFFFFFFFu;
    constexpr uint32_t kRejected = 0xFFFFFFFEu; // Prim::prev: had a partner, but the move was not believable
    constexpr uint32_t kSynth = 0x80000000u;    // Prim::prev: index into the synthesised vertices
    inline bool hasPrev(uint32_t prev) { return prev < kRejected; }
    constexpr uint32_t kMaxFactor = 8u;
    // A corner further than this (GS pixels) outside the scissor rectangle is "far": most such
    // corners are made by clipping and are a different point of the triangle every frame.
    constexpr float kFarMargin = 96.0f;

    inline uint64_t nowNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    uint32_t envFactor()
    {
        const char *v = std::getenv("PS2_FRAME_INTERP");
        const long n = v ? std::atol(v) : 1;
        return static_cast<uint32_t>(std::clamp<long>(n, 1, static_cast<long>(kMaxFactor)));
    }

    // A matched vertex that moved further than this (GS pixels) between two frames is not
    // interpolated: it is far more likely a wrong match than real motion.
    float envMaxMove()
    {
        const char *v = std::getenv("PS2_FRAME_INTERP_MAXPX");
        const double n = v ? std::atof(v) : 96.0;
        return static_cast<float>(n > 0.0 ? n : 96.0);
    }

    std::atomic<uint32_t> s_wantFactor{0}; // 0 = not set yet (environment)
    std::atomic<uint64_t> s_nextDueNs{0};

    // GS memory in 8 KiB pages, and the pages a pixel rectangle of a buffer at block `bp` touches.
    using PageSet = std::bitset<512>;
    constexpr uint32_t kPageBytes = 8192u;

    void addRectPages(PageSet &set, uint32_t bp, uint32_t bw, uint8_t psm, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1)
    {
        if (x1 < x0 || y1 < y0)
            return;
        uint32_t pw = 64u, ph = 32u;
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            ph = 64u;
            break;
        case GS_PSM_T8:
            pw = 128u;
            ph = 64u;
            break;
        case GS_PSM_T4:
            pw = 128u;
            ph = 128u;
            break;
        default:
            break;
        }
        const uint32_t ppr = std::max<uint32_t>(1u, (std::max<uint32_t>(bw, 1u) * 64u + pw - 1u) / pw);
        const uint32_t base = bp >> 5;
        const bool spill = (bp & 31u) != 0u;
        const uint32_t r0 = y0 / ph, r1 = y1 / ph, c0 = x0 / pw, c1 = x1 / pw;
        if ((r1 - r0 + 1u) * ppr >= 512u)
        {
            set.set();
            return;
        }
        for (uint32_t r = r0; r <= r1; ++r)
            for (uint32_t c = c0; c <= c1; ++c)
            {
                const uint32_t page = base + r * ppr + c;
                set.set(page & 511u);
                if (spill)
                    set.set((page + 1u) & 511u);
            }
    }

    enum class OpKind : uint8_t
    {
        Prims,     // a = run index
        LoadClut,  // a = index into cluts
        Transfer,  // a = index into transfers
        Upload,    // a = offset into bytes, b = size
        Flush,
        TextureFlush,
        Readback   // a = index into readbacks
    };

    struct Op
    {
        OpKind kind;
        uint32_t a = 0, b = 0;
    };

    struct Run
    {
        uint32_t state = 0;
        uint32_t primStart = 0, primCount = 0;
        float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f; // what its vertices span
    };

    struct Prim
    {
        uint32_t vtx = 0;     // first vertex in Frame::verts
        uint32_t prev = kNone; // first vertex of the matching primitive in the previous frame
        uint32_t kick = 0;    // position in its object's output
        uint8_t count = 0;
        uint8_t uv = 0;       // 1 + index in Frame::uvModels: its texture is laid over the scene by a rule (see fitTextureModels)
    };

    struct Object
    {
        uint64_t key = 0;   // pc | (hash of the input data, mixed with the draw state) << 32
        uint64_t group = 0; // pc | draw state << 32
        float q0 = 0.0f;    // first vertex: perspective term and depth
        double z0 = 0.0;
        float dx = 0.0f, dy = 0.0f; // average movement of its matched primitives (to last frame)
        bool hasMotion = false;
        bool inScene = false;       // depth-tested against the 3D scene
        uint8_t motion = 0;         // 0 unknown, 1 moves with the world, 2 moves by itself, 3 turns with the camera at no distance (the sky)
        uint8_t prog = 0;           // index of its VU1 program in m_progPc
        float wx0 = 0.0f, wy0 = 0.0f, wx1 = 0.0f, wy1 = 0.0f; // the screen (plus a margin) in its coordinates
        bool zSlopeDone = false;
        float zSlope = 0.0f;        // depth per unit of q (see depthSlope)
        uint32_t primStart = 0, primCount = 0;
        float sumX = 0.0f, sumY = 0.0f; // of its vertices (for "which instance is which")
        uint32_t vertices = 0;
        uint32_t implausible = 0;       // primitives whose partner was not where they can have come from
        bool rejected = false;          // all of it keeps this frame's positions in the in-between pictures
        bool affineDone = false, affineOk = false;
        bool layer = false;   // a layer of the ground under it (see releaseWorldLayers)
        bool clipped = false; // cut up by the game's own clipping: its primitives are not last frame's (see fitCamera)
        float affine[6] = {};           // where it was: x' = a0 x + a1 y + a2, y' = a3 x + a4 y + a5 (see objectAffine)
        float ax0 = 0.0f, ay0 = 0.0f, ax1 = 0.0f, ay1 = 0.0f; // ... valid for corners in this rectangle

        float cx() const { return vertices ? sumX / static_cast<float>(vertices) : 0.0f; }
        float cy() const { return vertices ? sumY / static_cast<float>(vertices) : 0.0f; }
    };

    // A texture laid over the scene by a rule instead of by the model's own coordinates: the
    // coordinates are a linear function of where the vertex is in front of the camera,
    // (s/q, t/q) = l (x w, y w, w, 1) with w = 1 / q (see fitTextureModels).
    struct UvModel
    {
        uint32_t tbp0 = 0;
        double l[2][4]{};     // this frame's rule
        double lPrev[2][4]{}; // last frame's (for this frame's in-between pictures)
        float residual = 0.0f; // how well this frame's rule fits (texture repeats)
        bool valid = false;   // the texture follows a rule this frame
        bool hasPrev = false; // ... and last frame, and the two differ: the in-between pictures use both
    };

    struct ClutLoad
    {
        GSTex0Reg tex0;
        GSTexClutReg texclut;
    };

    struct Readback
    {
        uint32_t bytes = 0;
        std::function<void(std::vector<uint8_t> &&, uint32_t)> done;
        bool handled = false; // given to the renderer to read without waiting
    };

    // The renderer's ReadbackAsync hands over all of the transfer's bytes; the caller asked for `bytes`.
    std::function<void(std::vector<uint8_t> &&)> sizedReadback(std::function<void(std::vector<uint8_t> &&, uint32_t)> done, uint32_t bytes)
    {
        return [done = std::move(done), bytes](std::vector<uint8_t> &&data)
        {
            const uint32_t got = static_cast<uint32_t>(std::min<size_t>(data.size(), bytes));
            data.resize(bytes, 0u);
            done(std::move(data), got);
        };
    }

    struct XY
    {
        float x, y;
    };

    struct Frame
    {
        std::vector<GSDrawState> states;
        std::vector<GSVertex> verts;
        std::vector<XY> pos; // the vertices' positions again, packed (what the matching reads)
        std::vector<Prim> prims;
        std::vector<Run> runs;
        std::vector<Op> ops;
        std::vector<uint8_t> bytes;
        std::vector<GSTransferCommand> transfers;
        std::vector<ClutLoad> cluts;
        std::vector<Readback> readbacks;
        std::vector<Object> objects;
        std::vector<uint32_t> byKey; // object indices sorted by key (built at the flip)
        std::vector<GSVertex> synth; // last-frame positions made up for primitives without a partner
        std::vector<UvModel> uvModels; // textures laid over the scene by a rule
        uint64_t matchNs = 0, phaseNs[4]{}; // what the matching took
        bool haveModel = false;      // a camera model was found
        float modelShare = 0.0f;     // ... explaining this share of the sampled matched vertices
        size_t doneOps = 0;     // ops already executed (a call needed the renderer's state)
        bool broken = false;    // ... and they included drawing or transfers: no in-between pictures
        bool complete = false;  // recorded from flip to flip without a gap
        uint32_t matched = 0, rejected = 0, recovered = 0, carried = 0;
        uint32_t held = 0; // primitives of things that stay as a whole (see settleMovers)
        uint32_t layers = 0; // objects taken for layers of the ground under them (see releaseWorldLayers)

        void clear()
        {
            states.clear();
            verts.clear();
            pos.clear();
            prims.clear();
            runs.clear();
            ops.clear();
            bytes.clear();
            transfers.clear();
            cluts.clear();
            readbacks.clear();
            objects.clear();
            byKey.clear();
            synth.clear();
            uvModels.clear();
            matchNs = 0;
            phaseNs[0] = phaseNs[1] = phaseNs[2] = phaseNs[3] = 0;
            haveModel = false;
            modelShare = 0.0f;
            doneOps = 0;
            broken = false;
            complete = false;
            matched = rejected = recovered = carried = held = layers = 0;
        }
    };

    // The GS thread takes this once per primitive; other threads only for the rare calls that
    // need the renderer's state. Uncontended it is one atomic exchange.
    class SpinLock
    {
    public:
        void lock()
        {
            while (m_flag.exchange(true, std::memory_order_acquire))
            {
                while (m_flag.load(std::memory_order_relaxed))
                    std::this_thread::yield();
            }
        }
        void unlock() { m_flag.store(false, std::memory_order_release); }

    private:
        std::atomic<bool> m_flag{false};
    };

    class GsInterpBackend;
    GsInterpBackend *s_instance = nullptr;

    class GsInterpBackend final : public GSRasterBackend
    {
    public:
        GsInterpBackend(std::unique_ptr<GSRasterBackend> inner, GSRasterBackendEx *ex)
            : m_inner(std::move(inner)), m_ex(ex), m_maxMove(envMaxMove())
        {
            if (s_wantFactor.load() == 0u)
                s_wantFactor.store(envFactor());
            m_stats = std::getenv("PS2_GS_VK_STATS") != nullptr || std::getenv("PS2_FRAME_INTERP_STATS") != nullptr;
            m_active = s_wantFactor.load() > 1u;
            m_cur = acquireFrame();
            m_cur->complete = true;
            m_last = acquireFrame();
            // The passes run on a thread of their own (PS2_FRAME_INTERP_THREAD=0: on the GS thread).
            m_threaded = !(std::getenv("PS2_FRAME_INTERP_THREAD") && std::getenv("PS2_FRAME_INTERP_THREAD")[0] == '0');
            if (m_threaded)
                m_thread = std::thread(&GsInterpBackend::renderMain, this);
            if (const char *probe = std::getenv("PS2_FRAME_INTERP_PROBE"))
                m_probe = std::sscanf(probe, "%f,%f", &m_probeX, &m_probeY) == 2;
            s_instance = this;
            std::fprintf(stderr, "[gs:interp] frame interpolation layer: %u picture(s) per game frame (PS2_FRAME_INTERP=1..%u, F7 in game)\n",
                         s_wantFactor.load(), kMaxFactor);
        }
        ~GsInterpBackend() override
        {
            if (s_instance == this)
                s_instance = nullptr;
            if (m_thread.joinable())
            {
                {
                    std::lock_guard<std::mutex> lock(m_jobMutex);
                    m_quit = true;
                }
                m_jobCv.notify_all();
                m_thread.join();
            }
            // the frames go back to the pool before it goes
            m_job = Job{};
            m_cur.reset();
            m_last.reset();
        }

        void Initialize(uint8_t *vram, uint32_t vramSize) override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            drain();
            m_vram = vram;
            m_vramSize = vramSize;
            m_inner->Initialize(vram, vramSize);
        }
        void Reset() override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            drain();
            m_inner->Reset();
        }

        void Submit(const GSPrimitiveBatch &batch) override
        {
            if (!m_active)
            {
                if (m_haveHeld.load(std::memory_order_relaxed))
                    flushHeldLocked();
                m_inner->Submit(batch);
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            recordPrim(batch);
        }
        void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override
        {
            if (!m_active)
            {
                if (m_haveHeld.load(std::memory_order_relaxed))
                    flushHeldLocked();
                m_inner->LoadClut(tex0, texclut);
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            m_cur->cluts.push_back({tex0, texclut});
            pushOp(OpKind::LoadClut, static_cast<uint32_t>(m_cur->cluts.size() - 1u));
        }
        void BeginTransfer(const GSTransferCommand &command) override
        {
            if (!m_active)
            {
                if (m_haveHeld.load(std::memory_order_relaxed))
                    flushHeldLocked();
                if (command.direction == 1u && m_asyncReads)
                {
                    // A read-back: usually followed at once by deferReadback (the runtime's
                    // lens-flare probes), which lets the renderer read it without waiting.
                    // Anything else first, and it goes to the renderer as it is.
                    std::lock_guard<SpinLock> lock(m_mutex);
                    m_held = command;
                    m_haveHeld.store(true, std::memory_order_relaxed);
                    return;
                }
                m_inner->BeginTransfer(command);
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            m_cur->transfers.push_back(command);
            pushOp(OpKind::Transfer, static_cast<uint32_t>(m_cur->transfers.size() - 1u));
            m_transferBegun = true;
        }
        void UploadImage(const uint8_t *data, uint32_t sizeBytes) override
        {
            if (!m_active)
            {
                if (m_haveHeld.load(std::memory_order_relaxed))
                    flushHeldLocked();
                m_inner->UploadImage(data, sizeBytes);
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            // Image data of a transfer that began before the last flip: it can only be sent to
            // the renderer once, so this frame is drawn once.
            if (!m_transferBegun)
                m_cur->broken = true;
            const size_t at = m_cur->bytes.size();
            m_cur->bytes.insert(m_cur->bytes.end(), data, data + sizeBytes);
            pushOp(OpKind::Upload, static_cast<uint32_t>(at), sizeBytes);
        }
        void Flush() override
        {
            if (!m_active)
            {
                m_inner->Flush();
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            if (m_cur->ops.empty() || m_cur->ops.back().kind != OpKind::Flush)
                pushOp(OpKind::Flush);
        }
        void TextureFlush() override
        {
            if (!m_active)
            {
                m_inner->TextureFlush();
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            pushOp(OpKind::TextureFlush);
        }
        void Sync(GSSyncReason reason) override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            drain();
            m_inner->Sync(reason);
        }
        PresentationFrame Present(const GSPresentationRequest &request) override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            drain();
            return m_inner->Present(request);
        }
        void QueuePresentSnapshot(const GSPresentationRequest &request) override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            flip(request);
        }
        bool PresentsOnGpu() const override { return m_inner->PresentsOnGpu(); }
        void SetResolutionScale(uint32_t scale) override { m_inner->SetResolutionScale(scale); }
        uint32_t ResolutionScale() const override { return m_inner->ResolutionScale(); }

        bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            drain();
            return m_inner->ClearFramebuffer(context, rgba);
        }
        uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            drain();
            return m_inner->ConsumeLocalToHostBytes(dst, maxBytes);
        }
        uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override
        {
            auto *self = const_cast<GsInterpBackend *>(this);
            std::lock_guard<SpinLock> lock(self->m_mutex);
            self->drain();
            return m_inner->ReadVram(psm, base, bw, x, y);
        }
        void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            drain();
            m_inner->WriteVram(psm, base, bw, x, y, value);
        }
        void SnapshotVram(std::vector<uint8_t> &out) const override
        {
            auto *self = const_cast<GsInterpBackend *>(this);
            std::lock_guard<SpinLock> lock(self->m_mutex);
            self->drain();
            m_inner->SnapshotVram(out);
        }
        // Debug history only: not worth executing the buffered frame for.
        GSTransferSnapshot GetTransferSnapshot() const override { return m_inner->GetTransferSnapshot(); }

        void objectTag(uint32_t pc, uint32_t hash)
        {
            if (!m_active)
                return;
            std::lock_guard<SpinLock> lock(m_mutex);
            beginObject(static_cast<uint64_t>(pc) | (static_cast<uint64_t>(hash) << 32));
        }

        bool deferReadback(uint32_t bytes, std::function<void(std::vector<uint8_t> &&, uint32_t)> done)
        {
            if (!m_active)
            {
                // Not buffering: the transfer just set up, read without waiting if the renderer can.
                std::lock_guard<SpinLock> lock(m_mutex);
                if (!m_haveHeld.load(std::memory_order_relaxed))
                    return false;
                m_haveHeld.store(false, std::memory_order_relaxed);
                if (innerReadbackAsync(m_held, sizedReadback(std::move(done), bytes)))
                    return true;
                m_inner->BeginTransfer(m_held);
                return false;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            m_cur->readbacks.push_back({bytes, std::move(done)});
            pushOp(OpKind::Readback, static_cast<uint32_t>(m_cur->readbacks.size() - 1u));
            return true;
        }

    private:
        bool innerReadbackAsync(const GSTransferCommand &command, std::function<void(std::vector<uint8_t> &&)> done)
        {
            return m_ex && m_ex->ReadbackAsync(command, std::move(done));
        }

        // The held read-back set-up (see BeginTransfer) goes to the renderer now.
        void flushHeld() // m_mutex held
        {
            if (m_haveHeld.load(std::memory_order_relaxed))
            {
                m_haveHeld.store(false, std::memory_order_relaxed);
                m_inner->BeginTransfer(m_held);
            }
        }
        void flushHeldLocked()
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            flushHeld();
        }

        void pushOp(OpKind kind, uint32_t a = 0, uint32_t b = 0)
        {
            m_cur->ops.push_back({kind, a, b});
            m_runOpen = false;
        }

        void beginObject(uint64_t key)
        {
            Frame &f = *m_cur;
            closeObject();
            Object obj;
            obj.key = key;
            obj.primStart = static_cast<uint32_t>(f.prims.size());
            {
                const uint32_t pc = static_cast<uint32_t>(key);
                size_t i = 0;
                while (i < m_progPc.size() && m_progPc[i] != pc)
                    ++i;
                if (i == m_progPc.size())
                {
                    if (m_progPc.size() < 255u)
                        m_progPc.push_back(pc);
                    else
                        i = 254u; // more programs than expected: share the last slot
                }
                obj.prog = static_cast<uint8_t>(i);
            }
            f.objects.push_back(obj);
            m_kickBase = g_ssx3GsKicks.load(std::memory_order_relaxed);
        }

        void closeObject()
        {
            Frame &f = *m_cur;
            if (f.objects.empty())
                return;
            Object &obj = f.objects.back();
            obj.primCount = static_cast<uint32_t>(f.prims.size()) - obj.primStart;
            if (obj.primCount == 0u)
                f.objects.pop_back();
        }

        void recordPrim(const GSPrimitiveBatch &batch)
        {
            Frame &f = *m_cur;
            if (batch.vertexCount == 0u || batch.vertexCount > 3u)
                return;
            if (f.objects.empty())
                beginObject(0xFFFFFFFFull); // nothing tagged yet in this frame
            if (!m_runOpen || std::memcmp(&f.states[f.runs.back().state], &batch.state, sizeof(GSDrawState)) != 0)
            {
                // The same state often comes back after an upload or CLUT load: reuse the last one.
                uint32_t stateIndex;
                if (!f.states.empty() && std::memcmp(&f.states.back(), &batch.state, sizeof(GSDrawState)) == 0)
                    stateIndex = static_cast<uint32_t>(f.states.size() - 1u);
                else
                {
                    f.states.push_back(batch.state);
                    stateIndex = static_cast<uint32_t>(f.states.size() - 1u);
                }
                Run run;
                run.state = stateIndex;
                run.primStart = static_cast<uint32_t>(f.prims.size());
                f.runs.push_back(run);
                f.ops.push_back({OpKind::Prims, static_cast<uint32_t>(f.runs.size() - 1u), 0u});
                m_runOpen = true;
            }
            Prim prim;
            prim.vtx = static_cast<uint32_t>(f.verts.size());
            prim.count = batch.vertexCount;
            prim.kick = static_cast<uint32_t>(g_ssx3GsKicks.load(std::memory_order_relaxed) - m_kickBase);
            f.verts.insert(f.verts.end(), batch.vertices.begin(), batch.vertices.begin() + batch.vertexCount);
            Object &obj = f.objects.back();
            if (obj.vertices == 0u)
            {
                // What kind of drawing this is (the parts of the state that do not change from
                // frame to frame): one model is often drawn in several layers - same vertices,
                // different blending, depth or texture format - and a layer must be paired with
                // the same layer of last frame, not with another one at the same place.
                const GSDrawState &st = batch.state;
                const GSContext &ctx = st.context;
                uint32_t sig = 2166136261u;
                auto mix = [&sig](uint32_t v)
                {
                    sig = (sig ^ v) * 16777619u;
                };
                mix(static_cast<uint32_t>(st.prim.type) | (st.prim.tme ? 0x100u : 0u) | (st.prim.abe ? 0x200u : 0u) | (st.prim.fge ? 0x400u : 0u) |
                    (st.prim.fst ? 0x800u : 0u) | (st.prim.iip ? 0x1000u : 0u));
                mix(static_cast<uint32_t>(ctx.alpha) & 0xFFu);     // not FIX: fades animate it
                mix(static_cast<uint32_t>(ctx.test) & 0x7F00Fu);   // not AREF
                mix(ctx.frame.fbp);
                mix(ctx.frame.fbmsk);
                mix(ctx.zbuf.zmask ? 1u : 0u);
                if (st.prim.tme)
                    mix(ctx.tex0.psm | (static_cast<uint32_t>(ctx.tex0.tw) << 8) | (static_cast<uint32_t>(ctx.tex0.th) << 16) |
                        (static_cast<uint32_t>(ctx.tex0.tfx) << 24) | (static_cast<uint32_t>(ctx.tex0.tcc) << 28));
                obj.group = (obj.key & 0xFFFFFFFFull) | (static_cast<uint64_t>(sig) << 32);
                obj.key ^= static_cast<uint64_t>(sig * 0x9E3779B1u) << 32;
                obj.q0 = batch.vertices[0].q;
                obj.z0 = batch.vertices[0].z;
                obj.inScene = ((ctx.test >> 16) & 1u) != 0u && ((ctx.test >> 17) & 3u) >= 2u;
                obj.wx0 = static_cast<float>(ctx.xyoffset.ofx >> 4) + static_cast<float>(ctx.scissor.x0) - kFarMargin;
                obj.wx1 = static_cast<float>(ctx.xyoffset.ofx >> 4) + static_cast<float>(ctx.scissor.x1) + kFarMargin;
                obj.wy0 = static_cast<float>(ctx.xyoffset.ofy >> 4) + static_cast<float>(ctx.scissor.y0) - kFarMargin;
                obj.wy1 = static_cast<float>(ctx.xyoffset.ofy >> 4) + static_cast<float>(ctx.scissor.y1) + kFarMargin;
            }
            Run &run = f.runs.back();
            for (uint32_t i = 0; i < prim.count; ++i)
            {
                const float x = batch.vertices[i].x, y = batch.vertices[i].y;
                f.pos.push_back({x, y});
                obj.sumX += x;
                obj.sumY += y;
                run.x0 = std::min(run.x0, x);
                run.x1 = std::max(run.x1, x);
                run.y0 = std::min(run.y0, y);
                run.y1 = std::max(run.y1, y);
            }
            obj.vertices += prim.count;
            f.prims.push_back(prim);
            ++run.primCount;
        }

        // Do two primitives come from the same triangle of the model? Their texture coordinates
        // say so: those belong to the model's vertices and do not change with the camera, while
        // a triangle the VU1 program clipped gets new vertices (and new coordinates).
        static inline bool sameSource(const GSVertex *c, const GSVertex *p, uint32_t count)
        {
            for (uint32_t v = 0; v < count; ++v)
            {
                if (c[v].u != p[v].u || c[v].v != p[v].v)
                    return false;
                // s/q and t/q equal within 0.2 % (compared without dividing)
                const float cq = c[v].q, pq = p[v].q;
                const float su = c[v].s * pq, pu = p[v].s * cq, sv = c[v].t * pq, pv = p[v].t * cq;
                const float tol = 0.002f * std::fabs(cq * pq);
                if (std::fabs(su - pu) > tol + 0.002f * std::fabs(pu) || std::fabs(sv - pv) > tol + 0.002f * std::fabs(pv))
                    return false;
            }
            return true;
        }

        // Could the primitive have moved from p to c in one frame? Not further than the limit,
        // and its shape (the edges from the first vertex) about the same. `mover`: wider limits,
        // for a triangle known to be the same one of the same model (same input data, same
        // place in the program's output or same texture coordinates) of something that moves
        // by itself - a rider passing close to the camera moves a hundred pixels a frame and
        // turns visibly, and is still the same rider; there the limits only have to catch what
        // cannot be moved in a straight line. (Not for what stands in the world: its far-moving
        // triangles are the ones next to the camera, half off the screen, and the camera model
        // places those better than their partners do.)
        inline bool plausibleMove(const XY *c, const XY *p, uint32_t count, bool mover) const
        {
            const bool certain = mover;
            const float limit = certain ? m_maxMove * 2.5f : m_maxMove;
            for (uint32_t v = 0; v < count; ++v)
                if (std::fabs(c[v].x - p[v].x) > limit || std::fabs(c[v].y - p[v].y) > limit)
                    return false;
            for (uint32_t v = 1; v < count; ++v)
            {
                const float cex = c[v].x - c[0].x, cey = c[v].y - c[0].y, pex = p[v].x - p[0].x, pey = p[v].y - p[0].y;
                const float longest = std::max(std::max(std::fabs(cex), std::fabs(cey)), std::max(std::fabs(pex), std::fabs(pey)));
                const float allowed = (certain ? 1.0f : 0.5f) * longest + 4.0f;
                if (std::fabs(cex - pex) > allowed || std::fabs(cey - pey) > allowed)
                    return false;
            }
            return true;
        }

        // Pair the primitives of object `co` (this frame) with those of `po` (last frame). Both
        // come out of the same program in the same order, except where triangles were clipped
        // (a clipped one becomes a different number of primitives): walk both lists and, where
        // they stop agreeing, look a few primitives ahead in either for the next agreement.
        void matchPrims(Object &co, const Object &po, bool sameData)
        {
            Frame &f = *m_cur;
            Prim *cp = f.prims.data() + co.primStart;
            const Prim *pp = m_last->prims.data() + po.primStart;
            const GSVertex *cverts = f.verts.data(), *pverts = m_last->verts.data();
            const XY *cpos = f.pos.data(), *ppos = m_last->pos.data();
            constexpr uint32_t kWindow = 8u;
            const uint32_t objectIndex = static_cast<uint32_t>(&co - f.objects.data());
            // a partner that is certainly the same triangle, but not where it can have come from
            // (kept: settleMovers takes it after all when the object moves by itself)
            auto implausible = [&](uint32_t prim, uint32_t partner)
            {
                ++f.rejected; // (may still be placed from its neighbours)
                if (sameData)
                {
                    ++co.implausible;
                    m_implausible.push_back({objectIndex, co.primStart + prim, partner,
                                             plausibleMove(cpos + cp[prim].vtx, ppos + partner, cp[prim].count, true)});
                }
            };
            // The usual case: nothing clipped, both lists have the same primitives at the same
            // places in the program's output - pair them in order. Only for the same model data:
            // a mesh the game builds anew every frame (shadows, tracks in the snow) can have the
            // same number of triangles and still be a different mesh, so its primitives are
            // compared one by one below.
            if (sameData && co.primCount == po.primCount)
            {
                bool identical = true;
                for (uint32_t k = 0; k < co.primCount && identical; ++k)
                    identical = cp[k].kick == pp[k].kick && cp[k].count == pp[k].count;
                if (identical)
                {
                    for (uint32_t k = 0; k < co.primCount; ++k)
                    {
                        if (plausibleMove(cpos + cp[k].vtx, ppos + pp[k].vtx, cp[k].count, false))
                        {
                            cp[k].prev = pp[k].vtx;
                            ++f.matched;
                        }
                        else
                            implausible(k, pp[k].vtx);
                    }
                    return;
                }
            }
            uint32_t i = 0, j = 0;
            auto same = [&](uint32_t a, uint32_t b)
            {
                return cp[a].count == pp[b].count && sameSource(cverts + cp[a].vtx, pverts + pp[b].vtx, cp[a].count);
            };
            while (i < co.primCount && j < po.primCount)
            {
                if (!same(i, j))
                {
                    bool found = false;
                    for (uint32_t d = 1; d <= 2u * kWindow && !found; ++d)
                    {
                        for (uint32_t di = d > kWindow ? d - kWindow : 0u; di <= std::min(d, kWindow); ++di)
                        {
                            const uint32_t dj = d - di;
                            if (i + di < co.primCount && j + dj < po.primCount && same(i + di, j + dj))
                            {
                                i += di;
                                j += dj;
                                found = true;
                                break;
                            }
                        }
                    }
                    if (!found)
                    {
                        ++i;
                        ++j;
                        continue;
                    }
                }
                if (plausibleMove(cpos + cp[i].vtx, ppos + pp[j].vtx, cp[i].count, false))
                {
                    cp[i].prev = pp[j].vtx;
                    ++f.matched;
                }
                else
                    implausible(i, pp[j].vtx);
                ++i;
                ++j;
            }
        }

        // An object that turns or deforms a lot in one frame (the trick meter's spinning coil,
        // some effects) cannot be moved in straight lines: its triangles pass through each other
        // and collapse on the way. Two signs of that, counted over its sizeable triangles:
        //  - triangles that all but vanish half-way while facing the same way before and after
        //    (turning by more than a right angle on the screen, or moving through each other):
        //    more than a tenth of them;
        //  - triangles that face the other way than last frame: more than a third of them. Some
        //    always do - the ones at the edge of anything that turns, a rider's arm, a board in
        //    a flip - and half-way they are thin there, as they should be; a tenth of them was
        //    enough to stop a body part of a rider doing a trick, while the parts next to it
        //    moved on, and the rider came apart in the in-between pictures.
        // Such an object keeps this frame's positions in the in-between pictures.
        // `wrongPartners`: counts as a sign of a new scene (a camera cut pairs the same models at
        // unrelated places); not when the object merely stays with its neighbours.
        void rejectObject(Object &o, bool wrongPartners)
        {
            Frame &f = *m_cur;
            Prim *cp = f.prims.data() + o.primStart;
            for (uint32_t i = 0; i < o.primCount; ++i)
            {
                if (hasPrev(cp[i].prev))
                {
                    --f.matched;
                    if (wrongPartners)
                        ++f.rejected;
                    else
                        ++f.held;
                }
                cp[i].prev = kRejected; // the whole object, also what found no partner
            }
            o.rejected = true;
            o.hasMotion = false;
        }

        void checkObject(Object &co)
        {
            Frame &f = *m_cur;
            Prim *cp = f.prims.data() + co.primStart;
            uint32_t triangles = 0, flipped = 0, collapsing = 0;
            for (uint32_t i = 0; i < co.primCount; ++i)
            {
                if (!hasPrev(cp[i].prev) || cp[i].count != 3u)
                    continue;
                const XY *c = f.pos.data() + cp[i].vtx;
                const XY *q = m_last->pos.data() + cp[i].prev;
                const float ac = (c[1].x - c[0].x) * (c[2].y - c[0].y) - (c[2].x - c[0].x) * (c[1].y - c[0].y);
                const float ap = (q[1].x - q[0].x) * (q[2].y - q[0].y) - (q[2].x - q[0].x) * (q[1].y - q[0].y);
                const float smaller = std::min(std::fabs(ac), std::fabs(ap));
                if (smaller < 4.0f)
                    continue; // slivers flip and vanish all the time
                ++triangles;
                if ((ac > 0.0f) != (ap > 0.0f))
                {
                    ++flipped;
                    continue;
                }
                const float mx0 = (c[0].x + q[0].x) * 0.5f, my0 = (c[0].y + q[0].y) * 0.5f;
                const float am = ((c[1].x + q[1].x) * 0.5f - mx0) * ((c[2].y + q[2].y) * 0.5f - my0) - ((c[2].x + q[2].x) * 0.5f - mx0) * ((c[1].y + q[1].y) * 0.5f - my0);
                if (std::fabs(am) < 0.25f * smaller)
                    ++collapsing;
            }
            static const bool s_dbg = std::getenv("PS2_FRAME_INTERP_DEBUG") != nullptr;
            if (s_dbg && (flipped + collapsing != 0u || std::getenv("PS2_FRAME_INTERP_DEBUG")[0] == '2'))
                std::fprintf(stderr, "[gs:interp] object %llx at %.0f,%.0f: %u prims, %u sizeable matched triangles, %u facing the other way, %u collapsing\n",
                             static_cast<unsigned long long>(co.key), co.cx(), co.cy(), co.primCount, triangles, flipped, collapsing);
            if ((collapsing >= 3u && collapsing * 10u > triangles) || (flipped >= 3u && flipped * 3u > triangles))
                rejectObject(co, true);
        }

        // Things that move by themselves (riders, boards, effects), after the camera model has
        // said which those are:
        //  - their primitives whose partner was too far away or too different for the usual
        //    limits are followed after all, within the wider limits (see plausibleMove). They
        //    used to be placed by the object's average movement or - when none of its primitives
        //    could be followed - by the camera model, as if the thing stood still in the world:
        //    a rider next to the camera came apart in the in-between pictures;
        //  - one that still has primitives that cannot be followed stays as a whole;
        //  - a rider is drawn as dozens of objects, a body part each (and each part in several
        //    layers). Half a rider moved on and half not is a rider torn apart, so when a good
        //    part of one stays (a sixth of its primitives), all of it does. What belongs together
        //    is known from the corners: neighbouring parts have corners at the very same place.
        // The world's layers. The ground is drawn in several passes over the very same corners
        // - base texture, detail, shadows, each by its own VU1 program. The passes are matched
        // one by one, and a pass can come out differently from the one under it: its triangles
        // paired with the wrong ones of last frame (the game clips some passes itself, see
        // fitCamera), it then looks like a thing that moves by itself, cannot be followed, and
        // stays where it is while the pass under it moves on. In the in-between picture the
        // layers then lie apart: a strip of ground without its base, the sky showing through.
        // So: an object that is not known to move with the world, most of whose corners are
        // corners of followed primitives that do, is a layer of that ground. Its own matches
        // are dropped; recoverUnmatched gives every corner the position its twin has.
        void releaseWorldLayers()
        {
            Frame &f = *m_cur;
            m_layerObjects.clear();
            size_t corners = 0;
            for (uint32_t k = 0; k < f.objects.size(); ++k)
            {
                const Object &o = f.objects[k];
                if (!o.inScene || o.primCount == 0u || (o.motion == 1u && !o.rejected) || o.motion == 3u)
                    continue;
                m_layerObjects.push_back(k);
                corners += o.vertices;
            }
            if (m_layerObjects.empty())
                return;
            uint32_t capacity = 256u;
            while (capacity < corners * 2u)
                capacity <<= 1;
            m_layerTable.assign(capacity, CornerEntry{});
            const auto slotFor = [&](const XY &at, uint64_t &key)
            {
                uint32_t xb, yb;
                std::memcpy(&xb, &at.x, 4);
                std::memcpy(&yb, &at.y, 4);
                key = (static_cast<uint64_t>(xb) << 32) | yb;
                uint32_t slot = static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> 40) & (capacity - 1u);
                while (m_layerTable[slot].used && m_layerTable[slot].key != key)
                    slot = (slot + 1u) & (capacity - 1u);
                return slot;
            };
            for (uint32_t k : m_layerObjects)
            {
                const Object &o = f.objects[k];
                const Prim *p = f.prims.data() + o.primStart;
                for (uint32_t q = 0; q < o.primCount; ++q)
                    for (uint32_t v = 0; v < p[q].count; ++v)
                    {
                        uint64_t key;
                        const uint32_t slot = slotFor(f.pos[p[q].vtx + v], key);
                        if (!m_layerTable[slot].used)
                            m_layerTable[slot] = CornerEntry{key, 0u, true}; // (mover: 1 once a world corner is there)
                    }
            }
            // the corners of what moves with the world and is followed
            for (const Object &o : f.objects)
            {
                if (!o.inScene || o.motion != 1u || o.rejected)
                    continue;
                const Prim *p = f.prims.data() + o.primStart;
                for (uint32_t q = 0; q < o.primCount; ++q)
                {
                    if (!hasPrev(p[q].prev))
                        continue;
                    for (uint32_t v = 0; v < p[q].count; ++v)
                    {
                        uint64_t key;
                        const uint32_t slot = slotFor(f.pos[p[q].vtx + v], key);
                        if (m_layerTable[slot].used)
                            m_layerTable[slot].mover = 1u;
                    }
                }
            }
            static const bool s_dbg = std::getenv("PS2_FRAME_INTERP_DEBUG") != nullptr;
            for (uint32_t k : m_layerObjects)
            {
                Object &o = f.objects[k];
                Prim *p = f.prims.data() + o.primStart;
                uint32_t all = 0, shared = 0;
                for (uint32_t q = 0; q < o.primCount; ++q)
                    for (uint32_t v = 0; v < p[q].count; ++v)
                    {
                        uint64_t key;
                        shared += m_layerTable[slotFor(f.pos[p[q].vtx + v], key)].mover;
                        ++all;
                    }
                if (all < 3u || shared * 5u < all * 3u)
                    continue;
                if (s_dbg)
                    std::fprintf(stderr, "[gs:interp] object %llx at %.0f,%.0f (%u prims, motion %u%s): %u of its %u corners are the moving ground's -> a layer of it\n",
                                 static_cast<unsigned long long>(o.key), o.cx(), o.cy(), o.primCount, o.motion, o.rejected ? ", was staying" : "", shared, all);
                for (uint32_t q = 0; q < o.primCount; ++q)
                {
                    if (hasPrev(p[q].prev))
                        --f.matched;
                    p[q].prev = kNone;
                }
                o.motion = 1u;
                o.hasMotion = false;
                o.rejected = false;
                o.layer = true;
                ++f.layers;
            }
        }

        void settleMovers()
        {
            Frame &f = *m_cur;
            releaseWorldLayers();
            for (size_t i = 0; i < m_implausible.size();)
            {
                const size_t first = i;
                const uint32_t index = m_implausible[i].object;
                Object &o = f.objects[index];
                while (i < m_implausible.size() && m_implausible[i].object == index)
                    ++i;
                // by itself: its followed primitives say so, or none could be followed - and what
                // its VU1 program draws does not as a rule move with the world (riders' programs:
                // by themselves; the ground's: with the world. The ground next to the camera,
                // half off the screen, has plenty of triangles that cannot be followed and a few
                // that seem to move by themselves; it is left to the camera model as before)
                const uint8_t progMotion = o.prog < m_progFrame.size() ? m_progFrame[o.prog].motion : uint8_t{0};
                const bool mover = !o.clipped && progMotion != 1u && (o.motion == 2u || (o.motion == 0u && progMotion == 2u));
                if (o.rejected || !mover)
                    continue;
                uint32_t failed = 0, taken = 0;
                float dx = 0.0f, dy = 0.0f;
                for (size_t k = first; k < i; ++k)
                {
                    Prim &p = f.prims[m_implausible[k].prim];
                    if (!m_implausible[k].wide || p.prev != kNone)
                    {
                        ++failed;
                        continue;
                    }
                    p.prev = m_implausible[k].partner;
                    ++f.matched;
                    --f.rejected;
                    dx += m_last->pos[p.prev].x - f.pos[p.vtx].x;
                    dy += m_last->pos[p.prev].y - f.pos[p.vtx].y;
                    ++taken;
                }
                if (taken != 0u)
                {
                    if (!o.hasMotion)
                    {
                        o.dx = dx / static_cast<float>(taken);
                        o.dy = dy / static_cast<float>(taken);
                        o.hasMotion = true;
                    }
                    o.motion = 2u;
                    checkObject(o); // (with the ones just taken)
                }
                if (!o.rejected && (failed >= 3u || failed * 10u > o.primCount))
                    rejectObject(o, false);
            }

            // The things in the scene that stay, and the ones that move by themselves around
            // them (within kAround of one that stays: a rider's parts are closer together than
            // that, and there is no need to look at every rider on the screen).
            constexpr float kAround = 320.0f;
            m_movers.clear();
            for (uint32_t k = 0; k < f.objects.size(); ++k)
            {
                const Object &o = f.objects[k];
                if (o.rejected && o.inScene && o.primCount != 0u)
                    m_movers.push_back(k);
            }
            if (m_movers.empty())
                return;
            const size_t held = m_movers.size();
            for (uint32_t k = 0; k < f.objects.size(); ++k)
            {
                const Object &o = f.objects[k];
                if (o.rejected || !o.inScene || o.primCount == 0u || o.motion != 2u)
                    continue;
                const float x = o.cx(), y = o.cy();
                for (size_t h = 0; h < held; ++h)
                {
                    const Object &r = f.objects[m_movers[h]];
                    const float dx = r.cx() - x, dy = r.cy() - y;
                    if (dx * dx + dy * dy <= kAround * kAround)
                    {
                        m_movers.push_back(k);
                        break;
                    }
                }
            }
            if (m_movers.size() == held)
                return;
            // joined by corners at the same place (union-find over m_movers)
            m_moverParent.resize(m_movers.size());
            for (uint32_t k = 0; k < m_moverParent.size(); ++k)
                m_moverParent[k] = k;
            auto find = [&](uint32_t k)
            {
                while (m_moverParent[k] != k)
                    k = m_moverParent[k] = m_moverParent[m_moverParent[k]];
                return k;
            };
            size_t corners = 0;
            for (uint32_t k : m_movers)
                corners += f.objects[k].vertices;
            uint32_t capacity = 256u;
            while (capacity < corners * 2u)
                capacity <<= 1;
            m_cornerTable.assign(capacity, CornerEntry{});
            for (uint32_t m = 0; m < m_movers.size(); ++m)
            {
                const Object &o = f.objects[m_movers[m]];
                const Prim *p = f.prims.data() + o.primStart;
                for (uint32_t q = 0; q < o.primCount; ++q)
                    for (uint32_t v = 0; v < p[q].count; ++v)
                    {
                        const XY &at = f.pos[p[q].vtx + v];
                        uint32_t xb, yb;
                        std::memcpy(&xb, &at.x, 4);
                        std::memcpy(&yb, &at.y, 4);
                        const uint64_t key = (static_cast<uint64_t>(xb) << 32) | yb;
                        uint32_t slot = static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> 40) & (capacity - 1u);
                        while (m_cornerTable[slot].used && m_cornerTable[slot].key != key)
                            slot = (slot + 1u) & (capacity - 1u);
                        if (!m_cornerTable[slot].used)
                            m_cornerTable[slot] = CornerEntry{key, m, true};
                        else
                        {
                            const uint32_t a = find(m), b = find(m_cornerTable[slot].mover);
                            if (a != b)
                                m_moverParent[a] = b;
                        }
                    }
            }
            // per group: primitives, and primitives that stay
            m_groupPrims.assign(m_movers.size(), 0u);
            m_groupHeld.assign(m_movers.size(), 0u);
            for (uint32_t m = 0; m < m_movers.size(); ++m)
            {
                const Object &o = f.objects[m_movers[m]];
                const uint32_t root = find(m);
                m_groupPrims[root] += o.primCount;
                m_groupHeld[root] += o.rejected ? o.primCount : 0u;
            }
            static const bool s_dbg = std::getenv("PS2_FRAME_INTERP_DEBUG") != nullptr;
            for (uint32_t m = 0; m < m_movers.size(); ++m)
            {
                const uint32_t root = find(m);
                if (s_dbg && root == m && m_groupHeld[root] != 0u)
                    std::fprintf(stderr, "[gs:interp] moving thing around %.0f,%.0f: %u of its %u primitives stay -> %s\n", f.objects[m_movers[m]].cx(), f.objects[m_movers[m]].cy(),
                                 m_groupHeld[root], m_groupPrims[root], m_groupHeld[root] * 6u >= m_groupPrims[root] ? "all of it stays" : "the rest moves");
                Object &o = f.objects[m_movers[m]];
                if (!o.rejected && m_groupHeld[root] * 6u >= m_groupPrims[root])
                    rejectObject(o, false);
            }
        }

        // Where a thing that moves by itself was last frame, as one movement of the whole object:
        // x' = a0 x + a1 y + a2, y' = a3 x + a4 y + a5, fitted to its followed primitives. For its
        // primitives without a partner (clipped differently at the edge of the screen). The
        // average movement alone, used before, is off by a lot at the far end of something
        // that turns close to the camera. False: too little to fit to - use the average.
        bool objectAffine(Object &o)
        {
            if (o.affineDone)
                return o.affineOk;
            o.affineDone = true;
            Frame &f = *m_cur;
            const Prim *p = f.prims.data() + o.primStart;
            const double ox = o.cx(), oy = o.cy();
            double sxx = 0, sxy = 0, syy = 0, sx = 0, sy = 0, tx[3] = {}, ty[3] = {};
            float bx0 = 1e30f, by0 = 1e30f, bx1 = -1e30f, by1 = -1e30f;
            uint32_t n = 0;
            for (uint32_t k = 0; k < o.primCount; ++k)
            {
                if (!hasPrev(p[k].prev) || (p[k].prev & kSynth))
                    continue;
                for (uint32_t v = 0; v < p[k].count; ++v)
                {
                    const XY &c = f.pos[p[k].vtx + v];
                    if (c.x < o.wx0 || c.x > o.wx1 || c.y < o.wy0 || c.y > o.wy1)
                        continue;
                    const XY &q = m_last->pos[p[k].prev + v];
                    const double x = c.x - ox, y = c.y - oy;
                    bx0 = std::min(bx0, c.x);
                    bx1 = std::max(bx1, c.x);
                    by0 = std::min(by0, c.y);
                    by1 = std::max(by1, c.y);
                    sxx += x * x;
                    sxy += x * y;
                    syy += y * y;
                    sx += x;
                    sy += y;
                    tx[0] += q.x * x;
                    tx[1] += q.x * y;
                    tx[2] += q.x;
                    ty[0] += q.y * x;
                    ty[1] += q.y * y;
                    ty[2] += q.y;
                    ++n;
                }
            }
            if (n < 9u)
                return false;
            // normal equations [sxx sxy sx; sxy syy sy; sx sy n] a = t, by Cramer's rule
            const double m[3][3] = {{sxx, sxy, sx}, {sxy, syy, sy}, {sx, sy, static_cast<double>(n)}};
            auto det3 = [](const double a[3][3])
            {
                return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
            };
            const double det = det3(m);
            // (spread in both directions: at least a few pixels across its narrow side)
            const double vx = sxx / n - (sx / n) * (sx / n), vy = syy / n - (sy / n) * (sy / n), vxy = sxy / n - (sx / n) * (sy / n);
            if (!(vx * vy - vxy * vxy > 4.0 * 4.0 * std::max(vx, vy)) || std::fabs(det) < 1e-9)
                return false;
            double a[6];
            for (int row = 0; row < 2; ++row)
            {
                const double *t = row == 0 ? tx : ty;
                for (int col = 0; col < 3; ++col)
                {
                    double r[3][3];
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            r[i][j] = j == col ? t[i] : m[i][j];
                    a[row * 3 + col] = det3(r) / det;
                }
            }
            // nothing wild: neither axis more than doubled or halved
            const double lx = std::sqrt(a[0] * a[0] + a[3] * a[3]), ly = std::sqrt(a[1] * a[1] + a[4] * a[4]);
            if (!(lx > 0.5 && lx < 2.0 && ly > 0.5 && ly < 2.0))
                return false;
            // (in the object's own coordinates so far: x - ox, y - oy)
            o.affine[0] = static_cast<float>(a[0]);
            o.affine[1] = static_cast<float>(a[1]);
            o.affine[2] = static_cast<float>(a[2] - a[0] * ox - a[1] * oy);
            o.affine[3] = static_cast<float>(a[3]);
            o.affine[4] = static_cast<float>(a[4]);
            o.affine[5] = static_cast<float>(a[5] - a[3] * ox - a[4] * oy);
            // (not far beyond the corners it was fitted to: half their extent on each side)
            o.ax0 = bx0 - 0.5f * (bx1 - bx0);
            o.ax1 = bx1 + 0.5f * (bx1 - bx0);
            o.ay0 = by0 - 0.5f * (by1 - by0);
            o.ay1 = by1 + 0.5f * (by1 - by0);
            o.affineOk = true;
            return true;
        }

        // ------------------------------------------------------------------------------------
        // The camera model. Everything that stands still in the game's world moves on screen by
        // one rule per frame: with w = 1 / q, the point (x w, y w, w) of last frame is a linear
        // function of (x w, y w, w, 1) of this frame (the camera's movement between the two
        // frames). The rule is fitted to matched vertices of unchanged models and then answers
        // "where was this point last frame" for vertices that have no partner: triangles that
        // were clipped differently, the corners clipping makes, things that just came into view
        // and meshes the game rebuilds every frame (tracks, shadows).
        // ------------------------------------------------------------------------------------
        struct Normal
        {
            double a[10]{}; // sum of b b^T (upper triangle)
            double r[3][4]{}; // sum of target_i b
            void add(const double b[4], const double t[3])
            {
                uint32_t k = 0;
                for (uint32_t i = 0; i < 4u; ++i)
                    for (uint32_t j = i; j < 4u; ++j)
                        a[k++] += b[i] * b[j];
                for (uint32_t i = 0; i < 3u; ++i)
                    for (uint32_t j = 0; j < 4u; ++j)
                        r[i][j] += t[i] * b[j];
            }
            bool solve(double h[3][4]) const
            {
                double m[4][7];
                uint32_t k = 0;
                for (uint32_t i = 0; i < 4u; ++i)
                    for (uint32_t j = i; j < 4u; ++j)
                        m[i][j] = m[j][i] = a[k++];
                for (uint32_t i = 0; i < 4u; ++i)
                    for (uint32_t c = 0; c < 3u; ++c)
                        m[i][4u + c] = r[c][i];
                for (uint32_t col = 0; col < 4u; ++col)
                {
                    uint32_t pivot = col;
                    for (uint32_t row = col + 1u; row < 4u; ++row)
                        if (std::fabs(m[row][col]) > std::fabs(m[pivot][col]))
                            pivot = row;
                    if (std::fabs(m[pivot][col]) < 1e-12)
                        return false;
                    if (pivot != col)
                        for (uint32_t c = 0; c < 7u; ++c)
                            std::swap(m[pivot][c], m[col][c]);
                    for (uint32_t row = 0; row < 4u; ++row)
                    {
                        if (row == col)
                            continue;
                        const double factor = m[row][col] / m[col][col];
                        for (uint32_t c = col; c < 7u; ++c)
                            m[row][c] -= factor * m[col][c];
                    }
                }
                for (uint32_t c = 0; c < 3u; ++c)
                    for (uint32_t i = 0; i < 4u; ++i)
                        h[c][i] = m[i][4u + c] / m[i][i];
                return true;
            }
        };

        struct Sample
        {
            float cx, cy, cq, px, py, pq;
            float cz;
            uint8_t prog;
        };

        struct ProgFrame
        {
            Normal normal;
            uint32_t samples = 0, inliers = 0;
            double sq = 0.0, sz = 0.0, sqq = 0.0, sqz = 0.0, szz = 0.0;
            uint32_t zn = 0;
            float zSlope = 0.0f;
            uint8_t motion = 0; // 0 unknown, 1 moves with the world, 2 moves by itself
        };

        static inline bool perspective(float q) { return q > 0.0f && q < 0.5f; }

        // Where the model `h` puts last frame's position of a vertex.
        static inline bool predictWith(const double h[3][4], float x, float y, float q, float &px, float &py, float &pq)
        {
            const double w = 1.0 / (1024.0 * static_cast<double>(q));
            const double b[4] = {(static_cast<double>(x) - 2048.0) / 256.0 * w, (static_cast<double>(y) - 2048.0) / 256.0 * w, w, 1.0};
            const double t0 = h[0][0] * b[0] + h[0][1] * b[1] + h[0][2] * b[2] + h[0][3];
            const double t1 = h[1][0] * b[0] + h[1][1] * b[1] + h[1][2] * b[2] + h[1][3];
            const double t2 = h[2][0] * b[0] + h[2][1] * b[1] + h[2][2] * b[2] + h[2][3];
            if (!(t2 > w * 0.125 && t2 < w * 8.0))
                return false; // behind the camera last frame, or nonsense
            px = static_cast<float>(t0 / t2 * 256.0 + 2048.0);
            py = static_cast<float>(t1 / t2 * 256.0 + 2048.0);
            pq = static_cast<float>(1.0 / (1024.0 * t2));
            return std::fabs(px - x) <= 2048.0f && std::fabs(py - y) <= 2048.0f;
        }

        static inline void sampleEquation(const Sample &s, double b[4], double t[3])
        {
            const double r = static_cast<double>(s.pq) / static_cast<double>(s.cq);
            b[0] = (static_cast<double>(s.cx) - 2048.0) / 256.0 * r;
            b[1] = (static_cast<double>(s.cy) - 2048.0) / 256.0 * r;
            b[2] = r;
            b[3] = 1024.0 * static_cast<double>(s.pq);
            t[0] = (static_cast<double>(s.px) - 2048.0) / 256.0;
            t[1] = (static_cast<double>(s.py) - 2048.0) / 256.0;
            t[2] = 1.0;
        }

        // Does the model explain the sample? Where it was (within 0.3 px plus a tenth of how
        // far it moved) and how far away it was (within a thousandth; what stands in the world
        // is within a hundred-thousandth). The distance matters: a rider standing about is
        // where the model says, near enough, but not as far away as it says, and four such
        // samples next to the camera among a thousand of the distant scenery were enough to
        // halve the camera's forward movement in the fit - nothing far away shows that, and
        // the ground under the camera, placed by the model, was off by up to twenty pixels.
        static inline bool agrees(const double h[3][4], const Sample &s, float slack)
        {
            float x, y, q;
            if (!predictWith(h, s.cx, s.cy, s.cq, x, y, q))
                return false;
            if (std::fabs(q - s.pq) > 0.001f * s.pq)
                return false;
            const float ex = x - s.px, ey = y - s.py;
            const float mx = s.cx - s.px, my = s.cy - s.py;
            const float allowed = slack + 0.1f * std::sqrt(mx * mx + my * my);
            return ex * ex + ey * ey <= allowed * allowed;
        }

        void fitCamera()
        {
            Frame &f = *m_cur;
            m_haveModel = false;
            m_samples.clear();
            m_progFrame.assign(m_progPc.size(), ProgFrame{});
            // What the game clips itself (the ground next to the camera, by a VU1 program of its
            // own) comes as a different set of triangles every frame: the n-th one of this frame
            // is not the n-th one of the last. The count gives it away - an object and its
            // partner differ in it - and so does the program, most of whose objects do. Such an
            // object's few "matched" primitives say nothing about how it moves, and it is never
            // taken for something that moves by itself (settleMovers, objectAffine).
            m_progClips.resize(m_progPc.size(), 0u);
            m_progPairs.assign(m_progPc.size() * 2u, 0u);
            for (const Pair &pair : m_pairs)
            {
                Object &o = f.objects[pair.cur];
                o.clipped = o.primCount != m_last->objects[pair.prev].primCount;
                if (o.prog < m_progPc.size())
                {
                    ++m_progPairs[2u * o.prog];
                    m_progPairs[2u * o.prog + 1u] += o.clipped ? 1u : 0u;
                }
            }
            for (size_t i = 0; i < m_progClips.size(); ++i)
            {
                if (m_progPairs[2u * i + 1u] >= 2u && m_progPairs[2u * i + 1u] * 4u >= m_progPairs[2u * i])
                    m_progClips[i] = 240u; // (frames it stays known for)
                else if (m_progClips[i] != 0u)
                    --m_progClips[i];
            }
            for (Object &o : f.objects)
                if (o.prog < m_progClips.size() && m_progClips[o.prog] != 0u)
                    o.clipped = true;
            uint32_t total = 0;
            for (const Pair &pair : m_pairs)
                if (pair.sameData)
                    total += f.objects[pair.cur].primCount;
            const uint32_t stride = std::max<uint32_t>(1u, total / 1500u);
            uint32_t counter = 0;
            for (const Pair &pair : m_pairs)
            {
                if (!pair.sameData)
                    continue;
                const Object &o = f.objects[pair.cur];
                const Prim *p = f.prims.data() + o.primStart;
                for (uint32_t k = 0; k < o.primCount; ++k, ++p)
                {
                    if (++counter % stride != 0u || !hasPrev(p->prev))
                        continue;
                    const uint32_t v = counter % p->count;
                    const XY &at = f.pos[p->vtx + v];
                    if (at.x < o.wx0 || at.x > o.wx1 || at.y < o.wy0 || at.y > o.wy1)
                        continue; // far corners are mostly made by clipping
                    const GSVertex &c = f.verts[p->vtx + v];
                    const GSVertex &q = m_last->verts[p->prev + v];
                    if (!perspective(c.q) || !perspective(q.q))
                        continue;
                    Sample s{c.x, c.y, c.q, q.x, q.y, q.q, static_cast<float>(c.z), o.prog};
                    m_samples.push_back(s);
                    ProgFrame &pf = m_progFrame[o.prog];
                    double b[4], t[3];
                    sampleEquation(s, b, t);
                    pf.normal.add(b, t);
                    ++pf.samples;
                    if (c.z > 0.0 && c.z < 16000000.0)
                    {
                        pf.sq += c.q;
                        pf.sz += c.z;
                        pf.sqq += static_cast<double>(c.q) * c.q;
                        pf.sqz += static_cast<double>(c.q) * c.z;
                        pf.szz += c.z * c.z;
                        ++pf.zn;
                    }
                }
            }
            if (m_samples.size() < 64u)
                return;
            // Candidates: the model of each of the programs with the most samples. The one that
            // most of the frame agrees with is the camera (things that move by themselves - the
            // riders - have models of their own, which little else follows).
            uint32_t order[4] = {0, 0, 0, 0};
            uint32_t candidates = 0;
            for (uint32_t round = 0; round < 4u; ++round)
            {
                uint32_t best = 0xFFFFFFFFu;
                for (uint32_t i = 0; i < m_progFrame.size(); ++i)
                {
                    if (m_progFrame[i].samples < 48u)
                        continue;
                    bool used = false;
                    for (uint32_t k = 0; k < candidates; ++k)
                        used = used || order[k] == i;
                    if (!used && (best == 0xFFFFFFFFu || m_progFrame[i].samples > m_progFrame[best].samples))
                        best = i;
                }
                if (best == 0xFFFFFFFFu)
                    break;
                order[candidates++] = best;
            }
            const size_t evalStride = std::max<size_t>(1u, m_samples.size() / 750u);
            uint32_t bestCount = 0, evaluated = 0;
            double bestH[3][4]{};
            for (uint32_t k = 0; k < candidates; ++k)
            {
                double h[3][4];
                if (!m_progFrame[order[k]].normal.solve(h))
                    continue;
                uint32_t count = 0, seen = 0;
                for (size_t i = 0; i < m_samples.size(); i += evalStride, ++seen)
                    count += agrees(h, m_samples[i], 0.3f) ? 1u : 0u;
                evaluated = seen;
                if (count > bestCount)
                {
                    bestCount = count;
                    std::memcpy(bestH, h, sizeof(bestH));
                }
            }
            if (bestCount < 24u || bestCount * 100u < evaluated * 35u)
                return;
            // Refit on everything that agrees - twice, the second time on what agrees with the
            // first refit - and note per program how much of it does.
            uint32_t inliers = 0;
            std::memcpy(m_model, bestH, sizeof(m_model));
            for (uint32_t round = 0; round < 2u; ++round)
            {
                Normal all;
                uint32_t count = 0;
                for (const Sample &s : m_samples)
                {
                    if (!agrees(m_model, s, 0.3f))
                        continue;
                    double b[4], t[3];
                    sampleEquation(s, b, t);
                    all.add(b, t);
                    ++count;
                }
                double h[3][4];
                if (count < 24u || !all.solve(h))
                    break;
                // (a refit that fewer samples agree with than with what it was made from is no better)
                uint32_t after = 0;
                for (const Sample &s : m_samples)
                    after += agrees(h, s, 0.3f) ? 1u : 0u;
                if (after < count)
                    break;
                std::memcpy(m_model, h, sizeof(m_model));
            }
            for (const Sample &s : m_samples)
                if (agrees(m_model, s, 0.3f))
                {
                    ++m_progFrame[s.prog].inliers;
                    ++inliers;
                }
            m_haveModel = true;
            m_modelShare = static_cast<float>(inliers) / static_cast<float>(m_samples.size());
            // The sky (and the mountains on the horizon) is a small shell around the camera,
            // drawn with a projection of its own: it turns with the camera and never comes
            // closer. Its rule is the camera's without the part that moves the camera.
            std::memcpy(m_skyModel, m_model, sizeof(m_skyModel));
            m_skyModel[0][3] = m_skyModel[1][3] = m_skyModel[2][3] = 0.0;
            // Depth per unit of q (to give a point placed by the model its old depth): per
            // program, and for programs without samples that of the program whose depth follows
            // q most closely (the world's projection).
            double bestFit = 0.0;
            m_zSlope = 0.0f;
            for (ProgFrame &pf : m_progFrame)
            {
                if (pf.samples >= 8u)
                    pf.motion = pf.inliers * 2u >= pf.samples ? 1u : 2u;
                const double det = pf.zn * pf.sqq - pf.sq * pf.sq;
                const double zvar = pf.zn * pf.szz - pf.sz * pf.sz;
                if (pf.zn < 8u || det <= 0.0 || zvar <= 0.0)
                    continue;
                const double cov = pf.zn * pf.sqz - pf.sq * pf.sz;
                pf.zSlope = static_cast<float>(cov / det);
                const double fit = cov * cov / (det * zvar);
                if (pf.motion == 1u && pf.zn >= 32u && fit > bestFit)
                {
                    bestFit = fit;
                    m_zSlope = pf.zSlope;
                }
            }
        }

        // Depth per unit of q for an object: from its own vertices when they say (a rebuilt mesh
        // has no matched samples), else its program's, else the world's.
        bool ownDepthSlope(const Object &o, float &slope) const
        {
            const Frame &f = *m_cur;
            double sq = 0.0, sz = 0.0, sqq = 0.0, sqz = 0.0, szz = 0.0;
            uint32_t n = 0;
            for (uint32_t k = 0; k < o.primCount; ++k)
            {
                const Prim &p = f.prims[o.primStart + k];
                for (uint32_t v = 0; v < p.count; ++v)
                {
                    const GSVertex &c = f.verts[p.vtx + v];
                    if (!perspective(c.q) || !(c.z > 0.0 && c.z < 16000000.0))
                        continue;
                    sq += c.q;
                    sz += c.z;
                    sqq += static_cast<double>(c.q) * c.q;
                    sqz += static_cast<double>(c.q) * c.z;
                    szz += c.z * c.z;
                    ++n;
                }
            }
            const double det = n * sqq - sq * sq, zvar = n * szz - sz * sz, cov = n * sqz - sq * sz;
            if (!(n >= 6u && det > 0.0 && zvar > 0.0 && cov * cov > 0.98 * det * zvar))
                return false;
            slope = static_cast<float>(cov / det);
            return true;
        }

        float depthSlope(Object &o)
        {
            if (!o.zSlopeDone)
            {
                o.zSlopeDone = true;
                float own = 0.0f;
                if (ownDepthSlope(o, own))
                    o.zSlope = own;
                else if (o.motion == 3u && m_skySlope != 0.0f)
                    o.zSlope = m_skySlope;
                else if (o.prog < m_progFrame.size() && m_progFrame[o.prog].zSlope != 0.0f)
                    o.zSlope = m_progFrame[o.prog].zSlope;
                else
                    o.zSlope = m_zSlope;
            }
            return o.zSlope;
        }

        // Is the matched far corner `c` (last frame `p`) the same point of the model in both
        // frames (then it is where the camera model says), or a corner made by clipping?
        inline bool farCornerIsReal(const GSVertex &c, const GSVertex &p, bool sky = false) const
        {
            float mx, my, mq;
            if (!predict(c, mx, my, mq, sky))
                return false;
            const float ex = mx - p.x, ey = my - p.y;
            const float allowed = 0.5f + 0.1f * (std::fabs(c.x - mx) + std::fabs(c.y - my));
            return ex * ex + ey * ey <= allowed * allowed;
        }

        // `sky`: for what turns with the camera at no distance (see fitCamera)
        inline bool predict(const GSVertex &c, float &x, float &y, float &q, bool sky = false) const
        {
            return m_haveModel && perspective(c.q) && predictWith(sky ? m_skyModel : m_model, c.x, c.y, c.q, x, y, q);
        }

        // Per paired object: how it moved on average, and whether it moves with the world
        // (the camera model explains it) or by itself. Then, for objects whose model data
        // changed (paired by kind and place, not by content):
        //  - moving with the world: a primitive whose partner is not where the camera model
        //    says is a wrong partner (the mesh was rebuilt in another order) - drop the match,
        //    the model places it;
        //  - part of the scene but drawn without perspective (the depth-only passes of a
        //    rebuilt mesh): nothing to check the partners with - drop the matches, the
        //    primitives take the old positions of the pass that has them.
        void classifyObjects()
        {
            Frame &f = *m_cur;
            for (const Pair &pair : m_pairs)
            {
                Object &o = f.objects[pair.cur];
                Prim *p = f.prims.data() + o.primStart;
                // (Every object is looked at, also an unchanged model of a program that all but
                // entirely follows the camera: the sky is drawn by the same program as most of the
                // scenery, and does not - it turns with the camera and never comes closer. Taken
                // for a part of the world, its pieces at the edge of the picture were placed by
                // the camera model, hundreds of pixels from where they belong: shards of sky
                // across the picture for one in-between picture.)
                // (what the game clips is looked at more closely: see below)
                const uint32_t step = std::max<uint32_t>(1u, o.primCount / (o.clipped ? 48u : 6u));
                uint32_t seen = 0, checked = 0, agree = 0, agreeSky = 0;
                float dx = 0.0f, dy = 0.0f;
                auto isNear = [&o](const XY &at)
                {
                    return at.x >= o.wx0 && at.x <= o.wx1 && at.y >= o.wy0 && at.y <= o.wy1;
                };
                for (uint32_t k = 0; k < o.primCount; k += step)
                {
                    if (!hasPrev(p[k].prev))
                        continue;
                    // a corner on or near the screen (far ones are mostly made by clipping)
                    uint32_t v = 0;
                    while (v < p[k].count && !isNear(f.pos[p[k].vtx + v]))
                        ++v;
                    if (v == p[k].count)
                        continue;
                    const GSVertex &c = f.verts[p[k].vtx + v];
                    const GSVertex &q = m_last->verts[p[k].prev + v];
                    dx += q.x - c.x;
                    dy += q.y - c.y;
                    ++seen;
                    if (m_haveModel && perspective(c.q) && perspective(q.q))
                    {
                        ++checked;
                        const Sample sample{c.x, c.y, c.q, q.x, q.y, q.q, 0.0f, 0u};
                        if (agrees(m_model, sample, 0.5f))
                            ++agree;
                        else if (agrees(m_skyModel, sample, 0.5f))
                            ++agreeSky;
                    }
                }
                if (seen != 0u)
                {
                    o.dx = dx / static_cast<float>(seen);
                    o.dy = dy / static_cast<float>(seen);
                    o.hasMotion = true;
                }
                // An object the game clipped (a different number of primitives than last frame)
                // had its primitives paired by their texture coordinates, and where those hardly
                // change across a triangle - the detail layer of the ground next to the camera -
                // a primitive can have been paired with its neighbour. None of those are where
                // the camera model says, and the rightly paired ones all are: a quarter of them
                // there is enough to know that the object stands in the world, and then every
                // partner is checked, also when the model data did not change. (Unchecked, the
                // detail layer kept its wrong partners while the base layer under it was placed
                // by the model: shards of ground across the picture for one in-between picture.)
                // The same for the sky, with its rule (what is far away follows both rules, and
                // counts as standing in the world).
                if (checked != 0u)
                {
                    const uint32_t share = o.clipped ? 4u : 2u;
                    o.motion = agree * share >= checked ? 1u : agreeSky * share >= checked ? 3u : 2u;
                }
                if ((pair.sameData && !o.clipped) || !m_haveModel)
                    continue;
                if (o.motion == 1u || o.motion == 3u)
                {
                    const double (*model)[4] = o.motion == 3u ? m_skyModel : m_model;
                    for (uint32_t k = 0; k < o.primCount; ++k)
                    {
                        if (!hasPrev(p[k].prev))
                            continue;
                        bool ok = true;
                        for (uint32_t v = 0; v < p[k].count && ok; ++v)
                        {
                            if (!isNear(f.pos[p[k].vtx + v]))
                                continue;
                            const GSVertex &c = f.verts[p[k].vtx + v];
                            const GSVertex &q = m_last->verts[p[k].prev + v];
                            ok = !perspective(c.q) || !perspective(q.q) || agrees(model, Sample{c.x, c.y, c.q, q.x, q.y, q.q, 0.0f, 0u}, 0.5f);
                        }
                        if (!ok)
                        {
                            p[k].prev = kNone;
                            --f.matched;
                        }
                    }
                }
                else if (o.motion == 0u && o.inScene && !pair.sameData)
                {
                    for (uint32_t k = 0; k < o.primCount; ++k)
                        if (hasPrev(p[k].prev))
                        {
                            p[k].prev = kNone;
                            --f.matched;
                        }
                    o.hasMotion = false;
                }
            }
            // Pieces of the sky without a partner (they come and go at the edge of the picture):
            // nothing says how they move, and taken for part of the world they were placed by
            // the camera's rule, far from where they belong - the sky jumped for one in-between
            // picture. The sky's projection gives them away: its depth per unit of q is a
            // thirtieth of the world's.
            if (!m_haveModel)
                return;
            double sum = 0.0;
            uint32_t n = 0;
            for (const Object &o : f.objects)
            {
                float slope = 0.0f;
                if (o.motion == 3u && ownDepthSlope(o, slope) && slope > 0.0f)
                {
                    sum += slope;
                    ++n;
                }
            }
            if (n != 0u)
                m_skySlope = static_cast<float>(sum / n);
            if (m_skySlope <= 0.0f || m_zSlope <= 0.0f || std::fabs(m_skySlope - m_zSlope) < 0.5f * m_zSlope)
                return;
            for (Object &o : f.objects)
            {
                float slope = 0.0f;
                if (o.motion == 0u && o.inScene && perspective(o.q0) && ownDepthSlope(o, slope) &&
                    std::fabs(slope - m_skySlope) < 0.2f * m_skySlope)
                    o.motion = 3u;
            }
        }

        // ------------------------------------------------------------------------------------
        // Textures laid over the scene by a rule. SSX 3 draws the glitter of the snow as one
        // more layer over the ground, by VU1 programs that work the texture coordinates out
        // from where each vertex is - (s/q, t/q) is a linear function of the vertex's place in
        // front of the camera, the same function for the whole frame and a different one every
        // frame: the pattern is not fixed to the ground, it drifts over it as the camera moves.
        // An in-between picture drew that layer with this frame's coordinates on vertices half
        // a frame back, so the pattern sat half a frame's drift away from where it belongs, in
        // every second picture shown: the glitter shook.
        // The rule is recovered per texture from the frame's own vertices (a texture whose
        // coordinates fit such a function all over the scene follows a rule: a model's own
        // coordinates do not, unless the model is flat - and then they do not change from
        // frame to frame, which is checked too) and kept with the frame. A vertex of an in-between picture then gets the coordinates
        // half-way between this frame's and those last frame's rule gives for where the vertex
        // was last frame.
        // ------------------------------------------------------------------------------------
        static inline void uvBasis(const GSVertex &c, double b[4])
        {
            const double w = 1.0 / (1024.0 * static_cast<double>(c.q));
            b[0] = (static_cast<double>(c.x) - 2048.0) / 256.0 * w;
            b[1] = (static_cast<double>(c.y) - 2048.0) / 256.0 * w;
            b[2] = w;
            b[3] = 1.0;
        }

        // calls fn(prim, model index) for the triangles of the scene's textured runs (`sample`:
        // for a part of them, enough to tell whether a texture follows a rule; `known`: only for
        // textures that have a rule)
        template <typename Fn>
        void forRuleCandidates(bool sample, bool known, Fn &&fn)
        {
            Frame &f = *m_cur;
            for (const Run &run : f.runs)
            {
                const GSDrawState &st = f.states[run.state];
                const GSContext &ctx = st.context;
                if (!st.prim.tme || st.prim.fst || !(((ctx.test >> 16) & 1u) != 0u && ((ctx.test >> 17) & 3u) >= 2u))
                    continue;
                uint8_t &slot = m_uvIndex[ctx.tex0.tbp0 & 0x3FFFu];
                if (slot == 0u)
                {
                    // (a texture found to follow no rule is not looked at again for a second)
                    if (known || f.uvModels.size() >= 250u || m_uvSkipUntil[ctx.tex0.tbp0 & 0x3FFFu] > m_matchCount)
                        continue;
                    f.uvModels.emplace_back();
                    f.uvModels.back().tbp0 = ctx.tex0.tbp0;
                    slot = static_cast<uint8_t>(f.uvModels.size());
                    m_uvIndexed.push_back(ctx.tex0.tbp0);
                }
                const uint32_t model = slot - 1u;
                if (known && !f.uvModels[model].valid)
                    continue;
                const uint32_t step = sample && run.primCount >= 24u ? 4u : 1u;
                for (uint32_t k = 0; k < run.primCount; k += step)
                {
                    Prim &p = f.prims[run.primStart + k];
                    if (p.count != 3u)
                        continue;
                    const GSVertex *c = f.verts.data() + p.vtx;
                    if (!perspective(c[0].q) || !perspective(c[1].q) || !perspective(c[2].q))
                        continue;
                    fn(p, model);
                }
            }
        }

        void fitTextureModels()
        {
            Frame &f = *m_cur;
            if (m_uvIndex.empty())
            {
                m_uvIndex.assign(0x4000u, 0u);
                m_uvSkipUntil.assign(0x4000u, 0ull);
            }
            for (uint32_t tbp0 : m_uvIndexed)
                m_uvIndex[tbp0 & 0x3FFFu] = 0u;
            m_uvIndexed.clear();
            f.uvModels.clear();
            m_uvNormals.clear();
            m_uvCounts.clear();
            forRuleCandidates(true, false, [&](Prim &p, uint32_t model)
            {
                if (m_uvNormals.size() <= model)
                {
                    m_uvNormals.resize(model + 1u);
                    m_uvCounts.resize(model + 1u, 0u);
                }
                const GSVertex *c = f.verts.data() + p.vtx;
                for (uint32_t v = 0; v < 3u; ++v)
                {
                    double b[4];
                    uvBasis(c[v], b);
                    const double t[3] = {static_cast<double>(c[v].s) / c[v].q, static_cast<double>(c[v].t) / c[v].q, 0.0};
                    m_uvNormals[model].add(b, t);
                }
                ++m_uvCounts[model];
            });
            bool any = false;
            for (uint32_t m = 0; m < f.uvModels.size(); ++m)
            {
                UvModel &u = f.uvModels[m];
                double h[3][4];
                if (m >= m_uvNormals.size() || m_uvCounts[m] < 16u || !m_uvNormals[m].solve(h))
                    continue;
                bool sane = true;
                for (uint32_t i = 0; i < 2u; ++i)
                    for (uint32_t j = 0; j < 4u; ++j)
                    {
                        u.l[i][j] = h[i][j];
                        sane = sane && std::fabs(h[i][j]) < 1e4;
                    }
                u.valid = sane;
            }
            // Do the sampled triangles follow their texture's rule (within a hundredth of a
            // repeat) - all but a twentieth of them? Else it is no rule, just a texture.
            constexpr double kTolerance = 0.01;
            const auto worstError = [&](const UvModel &u, const Prim &p)
            {
                const GSVertex *c = f.verts.data() + p.vtx;
                double worst = 0.0;
                for (uint32_t v = 0; v < 3u; ++v)
                {
                    double b[4];
                    uvBasis(c[v], b);
                    const double es = u.l[0][0] * b[0] + u.l[0][1] * b[1] + u.l[0][2] * b[2] + u.l[0][3] - static_cast<double>(c[v].s) / c[v].q;
                    const double et = u.l[1][0] * b[0] + u.l[1][1] * b[1] + u.l[1][2] * b[2] + u.l[1][3] - static_cast<double>(c[v].t) / c[v].q;
                    worst = std::max(worst, std::max(std::fabs(es), std::fabs(et)));
                }
                return worst;
            };
            m_uvFits.assign(f.uvModels.size(), 0u);
            forRuleCandidates(true, false, [&](Prim &p, uint32_t model)
            {
                const UvModel &u = f.uvModels[model];
                if (u.valid && worstError(u, p) <= kTolerance)
                    ++m_uvFits[model];
            });
            static const bool s_dbg = std::getenv("PS2_FRAME_INTERP_DEBUG") != nullptr;
            for (uint32_t m = 0; m < f.uvModels.size(); ++m)
            {
                UvModel &u = f.uvModels[m];
                if (u.valid && m_uvFits[m] * 20u < m_uvCounts[m] * 19u)
                    u.valid = false;
                if (!u.valid)
                    m_uvSkipUntil[u.tbp0 & 0x3FFFu] = m_matchCount + 48u + (u.tbp0 & 31u);
                any = any || u.valid;
                if (s_dbg && u.valid)
                    std::fprintf(stderr, "[gs:interp] texture %x is laid over the scene by a rule: %u of %u sampled triangles follow it\n", u.tbp0, m_uvFits[m], m_uvCounts[m]);
            }
            if (!any)
                return;
            // all triangles of those textures: which follow the rule
            forRuleCandidates(false, true, [&](Prim &p, uint32_t model)
            {
                UvModel &u = f.uvModels[model];
                const double worst = worstError(u, p);
                if (worst <= kTolerance)
                {
                    p.uv = static_cast<uint8_t>(model + 1u);
                    u.residual = std::max(u.residual, static_cast<float>(worst));
                }
            });
        }

        // Last frame's rule for each of this frame's, and whether the two differ at all. (After
        // recoverUnmatched: every primitive that moves has its last-frame vertices.)
        void pairTextureModels()
        {
            Frame &f = *m_cur;
            if (f.uvModels.empty() || !m_last)
                return;
            static const bool s_dbg = std::getenv("PS2_FRAME_INTERP_DEBUG") != nullptr;
            for (uint32_t m = 0; m < f.uvModels.size(); ++m)
            {
                UvModel &u = f.uvModels[m];
                if (!u.valid)
                    continue;
                const UvModel *before = nullptr;
                for (const UvModel &candidate : m_last->uvModels)
                    if (candidate.valid && candidate.tbp0 == u.tbp0)
                        before = &candidate;
                if (!before)
                    continue;
                std::memcpy(u.lPrev, before->l, sizeof(u.lPrev));
                // how far the pattern moved over the scene since last frame, on this frame's vertices
                double sum[2] = {0.0, 0.0}, largest = 0.0;
                uint32_t n = 0;
                m_uvChange.clear();
                for (const Prim &p : f.prims)
                {
                    if (p.uv != m + 1u || !hasPrev(p.prev))
                        continue;
                    const GSVertex *c = f.verts.data() + p.vtx;
                    const GSVertex *a = (p.prev & kSynth) ? f.synth.data() + (p.prev & ~kSynth) : m_last->verts.data() + p.prev;
                    if (!perspective(a[0].q))
                        continue;
                    double b[4];
                    uvBasis(a[0], b);
                    const double ds = u.lPrev[0][0] * b[0] + u.lPrev[0][1] * b[1] + u.lPrev[0][2] * b[2] + u.lPrev[0][3] - static_cast<double>(c[0].s) / c[0].q;
                    const double dt = u.lPrev[1][0] * b[0] + u.lPrev[1][1] * b[1] + u.lPrev[1][2] * b[2] + u.lPrev[1][3] - static_cast<double>(c[0].t) / c[0].q;
                    sum[0] += ds;
                    sum[1] += dt;
                    m_uvChange.push_back(static_cast<float>(ds));
                    m_uvChange.push_back(static_cast<float>(dt));
                    ++n;
                }
                if (n < 8u)
                    continue;
                // (a rule that jumped by whole repeats shows the same picture: taken out)
                const double whole[2] = {std::round(sum[0] / n), std::round(sum[1] / n)};
                u.lPrev[0][3] -= whole[0];
                u.lPrev[1][3] -= whole[1];
                for (size_t i = 0; i < m_uvChange.size(); i += 2u)
                    largest = std::max(largest, std::max(std::fabs(m_uvChange[i] - whole[0]), std::fabs(m_uvChange[i + 1u] - whole[1])));
                // a pattern that stays on the ground, or a jump that is no drift: this frame's coordinates
                static const bool s_off = std::getenv("PS2_FRAME_INTERP_NOUVRULE") != nullptr; // (debug: this frame's coordinates, as before)
                u.hasPrev = !s_off && largest > 5.0 * (static_cast<double>(u.residual) + before->residual) + 0.002 && largest < 4.0;
                if (s_dbg)
                    std::fprintf(stderr, "[gs:interp] texture %x: moved over the scene by up to %.4f repeats since last frame (%u triangles) -> %s\n", u.tbp0, largest, n,
                                 u.hasPrev ? "in-between pictures take the coordinates half-way" : "this frame's coordinates");
            }
        }

        // Two things would open cracks between neighbouring triangles in the in-between pictures:
        //  - primitives without a partner (mostly triangles the VU1 program clipped into a
        //    different number of pieces than last frame; also models that just appeared and
        //    meshes rebuilt every frame) would stay at this frame's position while their
        //    neighbours move;
        //  - the far-away corners that clipping makes sit on the edge of the GS coordinate range
        //    and are different points of the triangle every frame, so two triangles that share
        //    one disagree on where it was.
        // So such a corner gets its old position by the first of these that applies:
        //  1. it is at the same place as a corner that has one (a matched neighbour, the same
        //     triangle in another layer, a corner placed earlier): that one - exactly the same
        //     numbers, so layers drawn over each other stay over each other, depth included;
        //  2. the camera model, when the thing moves with the world;
        //  3. the average movement of its own object's matched primitives, when it moves by
        //     itself;
        //  4. for things in the 3D scene: the average movement of the matched primitives around
        //     it (64-pixel squares of GS coordinates).
        // What none of these places (new things drawn over the scene, like HUD digits) stays.
        void recoverUnmatched()
        {
            Frame &f = *m_cur;
            if (f.matched == 0u)
                return;
            // One pass over everything: the primitives to work on (no partner, or far corners,
            // or a matched depth-only pass), the motion squares, how many corners need a position.
            m_grid.assign(kGridCells * kGridCells, GridCell{});
            m_work.clear();
            uint32_t wanted = 0;
            size_t objectIndex = 0;
            for (const Run &run : f.runs)
            {
                const GSContext &ctx = f.states[run.state].context;
                const bool inScene = ((ctx.test >> 16) & 1u) != 0u && ((ctx.test >> 17) & 3u) >= 2u;
                const float wx0 = static_cast<float>(ctx.xyoffset.ofx >> 4) + static_cast<float>(ctx.scissor.x0) - kFarMargin;
                const float wx1 = static_cast<float>(ctx.xyoffset.ofx >> 4) + static_cast<float>(ctx.scissor.x1) + kFarMargin;
                const float wy0 = static_cast<float>(ctx.xyoffset.ofy >> 4) + static_cast<float>(ctx.scissor.y0) - kFarMargin;
                const float wy1 = static_cast<float>(ctx.xyoffset.ofy >> 4) + static_cast<float>(ctx.scissor.y1) + kFarMargin;
                const Prim *p = f.prims.data() + run.primStart;
                for (uint32_t k = 0; k < run.primCount; ++k, ++p)
                {
                    if (p->prev == kRejected)
                        continue;
                    const XY *c = f.pos.data() + p->vtx;
                    uint8_t farMask = 0u;
                    for (uint32_t v = 0; v < p->count; ++v)
                        if (c[v].x < wx0 || c[v].x > wx1 || c[v].y < wy0 || c[v].y > wy1)
                            farMask |= static_cast<uint8_t>(1u << v);
                    const uint32_t index = run.primStart + k;
                    const bool none = p->prev == kNone;
                    // matched, in the scene, drawn without perspective (a depth-only pass): it
                    // has to move exactly like the textured pass at the same place
                    const bool flat = !none && inScene && !perspective(f.verts[p->vtx].q);
                    if (none || farMask != 0u || flat)
                    {
                        while (objectIndex + 1u < f.objects.size() && index >= f.objects[objectIndex].primStart + f.objects[objectIndex].primCount)
                            ++objectIndex;
                        m_work.push_back({index, static_cast<uint32_t>(objectIndex), farMask, inScene, flat});
                        wanted += none || flat ? p->count : (farMask & 1u) + ((farMask >> 1) & 1u) + ((farMask >> 2) & 1u);
                    }
                    if (!none && !(farMask & 1u) && (index & 3u) == 0u) // (a sample is enough: the squares are the last resort)
                    {
                        const XY *q = m_last->pos.data() + p->prev;
                        GridCell &cell = m_grid[gridIndex(c[0].x, c[0].y)];
                        cell.dx += q[0].x - c[0].x;
                        cell.dy += q[0].y - c[0].y;
                        cell.n += 1.0f;
                    }
                }
            }
            if (wanted == 0u)
                return;
            // squares without matched primitives take the nearest square's motion
            m_gridQueue.clear();
            for (uint32_t i = 0; i < m_grid.size(); ++i)
            {
                GridCell &cell = m_grid[i];
                if (cell.n > 0.0f)
                {
                    cell.dx /= cell.n;
                    cell.dy /= cell.n;
                    m_gridQueue.push_back(i);
                }
            }
            for (size_t head = 0; head < m_gridQueue.size(); ++head)
            {
                const uint32_t i = m_gridQueue[head];
                const uint32_t cx = i % kGridCells, cy = i / kGridCells;
                const int32_t nx[4] = {static_cast<int32_t>(cx) - 1, static_cast<int32_t>(cx) + 1, static_cast<int32_t>(cx), static_cast<int32_t>(cx)};
                const int32_t ny[4] = {static_cast<int32_t>(cy), static_cast<int32_t>(cy), static_cast<int32_t>(cy) - 1, static_cast<int32_t>(cy) + 1};
                for (int k = 0; k < 4; ++k)
                {
                    if (nx[k] < 0 || ny[k] < 0 || nx[k] >= static_cast<int32_t>(kGridCells) || ny[k] >= static_cast<int32_t>(kGridCells))
                        continue;
                    GridCell &other = m_grid[static_cast<uint32_t>(ny[k]) * kGridCells + static_cast<uint32_t>(nx[k])];
                    if (other.n > 0.0f)
                        continue;
                    other.dx = m_grid[i].dx;
                    other.dy = m_grid[i].dy;
                    other.n = 1.0f;
                    m_gridQueue.push_back(static_cast<uint32_t>(ny[k]) * kGridCells + static_cast<uint32_t>(nx[k]));
                }
            }

            // The corners that need a position, by where they are - and by how far away they are
            // and whose they are: an entry per place, q and object. Layers of one surface must
            // get the very same old numbers, or they lie at different depths in the in-between
            // picture and fail each other's depth test. The patch of ground next door has a
            // corner at the same place too, this frame often with the very same numbers - but
            // not last frame (another VU1 program drew it then, and rounded differently). With
            // one entry per place, whichever came first served everyone: a layer placed from the
            // neighbour's old numbers under a layer that had its own partner - a white patch of
            // ground for one in-between picture. So a corner takes its own object's entry, else
            // that of the object drawn nearest to it (layers are drawn one after the other).
            uint32_t capacity = 64u;
            while (capacity < wanted * 4u)
                capacity <<= 1;
            m_table.assign(capacity, TableEntry{});
            const uint32_t mask = capacity - 1u;
            uint32_t tableUsed = 0;
            auto keyOf = [](const XY &v)
            {
                uint32_t xb, yb;
                std::memcpy(&xb, &v.x, 4);
                std::memcpy(&yb, &v.y, 4);
                return (static_cast<uint64_t>(xb) << 32) | yb;
            };
            auto firstSlot = [&](uint64_t key)
            {
                return static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> 40) & mask;
            };
            // the entry for that place, q and object, or the free slot where it would go
            auto slotOf = [&](const XY &v, float q, uint32_t object)
            {
                const uint64_t key = keyOf(v);
                uint32_t slot = firstSlot(key);
                while (m_table[slot].used && (m_table[slot].key != key || m_table[slot].qCur != q || m_table[slot].object != object))
                    slot = (slot + 1u) & mask;
                return slot;
            };
            auto insert = [&](uint32_t slot, const XY &v, float q, uint32_t object)
            {
                m_table[slot] = TableEntry{};
                m_table[slot].key = keyOf(v);
                m_table[slot].qCur = q;
                m_table[slot].object = object;
                m_table[slot].used = true;
                ++tableUsed;
            };
            auto want = [&](const XY &v, float q, uint32_t object)
            {
                const uint32_t slot = slotOf(v, q, object);
                if (!m_table[slot].used)
                    insert(slot, v, q, object);
            };
            // An entry: the old position of the point at that place (`prev`), how its distance
            // from the camera changed (`ratio`, old q / new q; 1 when unknown) and the depth of
            // the corner it came from. The first corner with those numbers that has an old
            // position fills it.
            auto offer = [&](const XY &at, uint32_t ref, const GSVertex &cur, const GSVertex &old, uint32_t object)
            {
                const uint32_t slot = slotOf(at, cur.q, object);
                if (!m_table[slot].used)
                {
                    if (tableUsed * 4u >= capacity * 3u)
                        return; // (full: the corners that want this place do without)
                    insert(slot, at, cur.q, object);
                }
                TableEntry &e = m_table[slot];
                if (e.prev != kNone)
                    return;
                const bool persp = perspective(cur.q) && perspective(old.q);
                e.prev = ref;
                e.persp = persp;
                e.ratio = persp ? old.q / cur.q : 1.0f;
                e.zCur = static_cast<float>(cur.z);
            };
            // The entry a corner takes its old position from: one with its own numbers (its own
            // object's, else the nearest object's), else - a neighbour's corner at the same
            // place - the one nearest in distance (within a hundredth: the game clips what
            // reaches past its drawing area to that area's edges, so corners of quite different
            // surfaces end up at the very same place, the area's corner above all). A corner
            // without perspective (a depth-only pass) takes one with perspective if there is
            // one, and one with perspective takes one without only when there is nothing else.
            auto lookup = [&](const XY &at, float q, uint32_t object) -> const TableEntry *
            {
                const uint64_t key = keyOf(at);
                const bool persp = perspective(q);
                const TableEntry *same = nullptr, *close = nullptr, *flat = nullptr;
                uint32_t sameDist = 0;
                float closeDiff = 0.0f;
                const auto distance = [object](const TableEntry &e)
                {
                    return e.object > object ? e.object - object : object - e.object;
                };
                for (uint32_t slot = firstSlot(key); m_table[slot].used; slot = (slot + 1u) & mask)
                {
                    const TableEntry &e = m_table[slot];
                    if (e.key != key || e.prev == kNone)
                        continue;
                    if (!e.persp)
                    {
                        if (!flat || (e.qCur == q && distance(e) < distance(*flat)))
                            flat = &e;
                        continue;
                    }
                    if (!persp || e.qCur == q)
                    {
                        const uint32_t d = distance(e);
                        if (!same || d < sameDist)
                        {
                            same = &e;
                            sameDist = d;
                        }
                        continue;
                    }
                    const float diff = std::fabs(q - e.qCur);
                    if (diff > 0.01f * e.qCur)
                        continue;
                    if (!close || diff < closeDiff)
                    {
                        close = &e;
                        closeDiff = diff;
                    }
                }
                return same ? same : close ? close : flat;
            };
            // is that place wanted at all (whatever the q)?
            auto wantedPlace = [&](const XY &at)
            {
                const uint64_t key = keyOf(at);
                for (uint32_t slot = firstSlot(key); m_table[slot].used; slot = (slot + 1u) & mask)
                    if (m_table[slot].key == key)
                        return true;
                return false;
            };
            // a small filter in front of the table: most corners of the frame are not wanted
            m_filter.assign(1024u, 0ull);
            auto filterBit = [&](const XY &v, uint32_t &word, uint64_t &bit)
            {
                const uint64_t h = keyOf(v) * 0x9E3779B97F4A7C15ull;
                word = static_cast<uint32_t>(h >> 54);
                bit = 1ull << ((h >> 48) & 63u);
            };
            for (const Work &w : m_work)
            {
                const Prim &p = f.prims[w.prim];
                for (uint32_t v = 0; v < p.count; ++v)
                    if (p.prev == kNone || w.flat || (w.farMask & (1u << v)))
                    {
                        want(f.pos[p.vtx + v], f.verts[p.vtx + v].q, w.object);
                        uint32_t word;
                        uint64_t bit;
                        filterBit(f.pos[p.vtx + v], word, bit);
                        m_filter[word] |= bit;
                    }
            }
            // ... and the ones that have one: the corners of matched primitives (a far corner
            // only if it is a real vertex: where the camera model says)
            {
                size_t cursor = 0;
                uint32_t object = 0;
                for (uint32_t index = 0; index < f.prims.size(); ++index)
                {
                    const Prim &p = f.prims[index];
                    while (cursor < m_work.size() && m_work[cursor].prim < index)
                        ++cursor;
                    while (object + 1u < f.objects.size() && index >= f.objects[object].primStart + f.objects[object].primCount)
                        ++object;
                    if (!hasPrev(p.prev))
                        continue;
                    const uint8_t farMask = cursor < m_work.size() && m_work[cursor].prim == index ? m_work[cursor].farMask : uint8_t{0};
                    for (uint32_t v = 0; v < p.count; ++v)
                    {
                        uint32_t word;
                        uint64_t bit;
                        filterBit(f.pos[p.vtx + v], word, bit);
                        if (!(m_filter[word] & bit))
                            continue;
                        if (!wantedPlace(f.pos[p.vtx + v]))
                            continue;
                        const GSVertex &c = f.verts[p.vtx + v];
                        const GSVertex &q = m_last->verts[p.prev + v];
                        if ((farMask & (1u << v)) && !farCornerIsReal(c, q, f.objects[object].motion == 3u))
                            continue;
                        offer(f.pos[p.vtx + v], p.prev + v, c, q, object);
                    }
                }
            }
            auto source = [&](uint32_t ref) -> const GSVertex &
            {
                return (ref & kSynth) ? f.synth[ref & ~kSynth] : m_last->verts[ref];
            };
            // takes the entry's old position for the corner `c`
            auto adopt = [&](const TableEntry &e, const GSVertex &c, GSVertex &old)
            {
                const GSVertex &pv = source(e.prev);
                old.x = pv.x;
                old.y = pv.y;
                // the same change of distance; the very same numbers for the same surface
                old.q = c.q == e.qCur ? pv.q : c.q * e.ratio;
                // The same depth as where it came from (a layer of the same surface): the same
                // old depth. Nearly the same (the neighbouring patch's corner, drawn by another
                // program): the same change of depth. Anything else (a pass at depth 0, say)
                // keeps its own.
                const double zEntry = static_cast<double>(e.zCur);
                if (static_cast<float>(c.z) == e.zCur)
                    old.z = pv.z;
                else if (c.z > 0.0 && zEntry > 0.0 && std::fabs(c.z - zEntry) <= 0.002 * zEntry + 2.0)
                    old.z = std::max(0.0, c.z + (static_cast<double>(pv.z) - zEntry));
                else
                    old.z = c.z;
            };

            // Round 0: what the table and the camera model place completely (and the far
            // corners of matched primitives). Round 1: the rest, with the weaker rules.
            for (uint32_t round = 0; round < 2u; ++round)
            {
                for (const Work &w : m_work)
                {
                    Prim &p = f.prims[w.prim];
                    const uint8_t farMask = w.farMask;
                    const bool inScene = w.inScene;
                    const bool matched = p.prev != kNone;
                    if (matched && (round != 0u || farMask == 0u || (p.prev & kSynth)))
                        continue;
                    Object &o = f.objects[w.object];
                    const uint8_t progMotion = o.prog < m_progFrame.size() ? m_progFrame[o.prog].motion : uint8_t{0};
                    // does it move with the world? its own matched primitives say, else its program, else: in the scene, yes
                    const bool withWorld = o.motion == 1u || (o.motion == 0u && (progMotion == 1u || (progMotion == 0u && inScene)));
                    const bool sky = o.motion == 3u;
                    const GSVertex *cv = f.verts.data() + p.vtx;
                    GSVertex old[3];
                    uint8_t placed = 0u; // corners with a position
                    uint8_t exact = 0u;  // ... from the table or the model
                    bool changed = false;
                    for (uint32_t v = 0; v < p.count; ++v)
                    {
                        const bool need = !matched || (farMask & (1u << v));
                        old[v] = matched ? m_last->verts[p.prev + v] : cv[v];
                        if (!need)
                        {
                            placed |= static_cast<uint8_t>(1u << v);
                            continue;
                        }
                        float mx = 0.0f, my = 0.0f, mq = 0.0f;
                        const bool model = sky ? predict(cv[v], mx, my, mq, true) : withWorld && predict(cv[v], mx, my, mq);
                        // a far corner of a matched primitive is only replaced when it is
                        // not where the camera model says (a corner made by clipping)
                        if (matched && (!model || farCornerIsReal(cv[v], old[v], sky)))
                            continue;
                        const TableEntry *entry = lookup(f.pos[p.vtx + v], cv[v].q, w.object);
                        if (entry)
                            adopt(*entry, cv[v], old[v]);
                        else if (model)
                        {
                            old[v].x = mx;
                            old[v].y = my;
                            old[v].q = mq;
                            old[v].z = cv[v].z > 0.0 ? std::max(0.0, cv[v].z + static_cast<double>(depthSlope(o)) * (static_cast<double>(mq) - static_cast<double>(cv[v].q))) : 0.0;
                        }
                        else
                            continue;
                        placed |= static_cast<uint8_t>(1u << v);
                        exact |= static_cast<uint8_t>(1u << v);
                        changed = true;
                    }
                    const uint8_t all = static_cast<uint8_t>((1u << p.count) - 1u);
                    if (matched)
                    {
                        if (!changed)
                            continue;
                    }
                    else if (placed != all)
                    {
                        if (round == 0u)
                            continue;
                        // the weaker rules for what is left
                        if (exact == 0u && !o.hasMotion && !inScene)
                            continue; // drawn over the scene, nothing known: stays
                        for (uint32_t v = 0; v < p.count; ++v)
                        {
                            if (placed & (1u << v))
                                continue;
                            if (o.hasMotion)
                            {
                                if (o.motion == 2u && !o.clipped && objectAffine(o) && old[v].x >= o.ax0 && old[v].x <= o.ax1 && old[v].y >= o.ay0 && old[v].y <= o.ay1)
                                {
                                    const float x = old[v].x, y = old[v].y;
                                    old[v].x = o.affine[0] * x + o.affine[1] * y + o.affine[2];
                                    old[v].y = o.affine[3] * x + o.affine[4] * y + o.affine[5];
                                }
                                else
                                {
                                    old[v].x += o.dx;
                                    old[v].y += o.dy;
                                }
                            }
                            else
                            {
                                const GridCell &cell = m_grid[gridIndex(cv[v].x, cv[v].y)];
                                old[v].x += cell.dx;
                                old[v].y += cell.dy;
                            }
                        }
                    }
                    const uint32_t base = static_cast<uint32_t>(f.synth.size());
                    f.synth.insert(f.synth.end(), old, old + p.count);
                    p.prev = kSynth | base;
                    // later corners at the same places take the same positions
                    for (uint32_t v = 0; v < p.count; ++v)
                        if (!matched || (farMask & (1u << v)))
                            offer(f.pos[p.vtx + v], kSynth | (base + v), cv[v], old[v], w.object);
                    if (!matched)
                    {
                        if (exact == all)
                            ++f.recovered;
                        else
                            ++f.carried;
                    }
                }
            }

            // Matched depth-only passes: the change of distance from the textured pass at the
            // same place, so that both move alike.
            for (const Work &w : m_work)
            {
                if (!w.flat)
                    continue;
                Prim &p = f.prims[w.prim];
                if (!hasPrev(p.prev) || (p.prev & kSynth))
                    continue;
                GSVertex old[3];
                bool changed = false;
                for (uint32_t v = 0; v < p.count; ++v)
                {
                    old[v] = m_last->verts[p.prev + v];
                    const TableEntry *found = lookup(f.pos[p.vtx + v], f.verts[p.vtx + v].q, w.object);
                    if (!found || !found->persp)
                        continue;
                    const TableEntry &e = *found;
                    const GSVertex &pv = source(e.prev);
                    if (std::fabs(pv.x - old[v].x) > 0.25f || std::fabs(pv.y - old[v].y) > 0.25f)
                        continue; // not the same thing after all
                    old[v].q = f.verts[p.vtx + v].q * e.ratio;
                    changed = true;
                }
                if (!changed)
                    continue;
                p.prev = kSynth | static_cast<uint32_t>(f.synth.size());
                f.synth.insert(f.synth.end(), old, old + p.count);
            }
        }

        static constexpr uint32_t kGridCells = 64u; // 64-pixel squares over the 4096 x 4096 GS coordinate space
        static inline uint32_t gridIndex(float x, float y)
        {
            const uint32_t cx = std::min<uint32_t>(kGridCells - 1u, static_cast<uint32_t>(std::max(0.0f, x)) >> 6);
            const uint32_t cy = std::min<uint32_t>(kGridCells - 1u, static_cast<uint32_t>(std::max(0.0f, y)) >> 6);
            return cy * kGridCells + cx;
        }

        // Pair the objects in cur[] and prev[] (indices into the frames' objects) by nearness on
        // screen: the closest pair first, each object used once, nothing further than the limit.
        // Could `po` be `co` one frame earlier, going by depth? The perspective term (1 / w) and
        // the depth of their first vertices must be of the same order: layers of one model drawn
        // without perspective or at depth 0 are different things, at the same place.
        static inline bool sameDepthRange(const Object &co, const Object &po)
        {
            if ((co.q0 > 0.0f) != (po.q0 > 0.0f))
                return false;
            if (co.q0 > 0.0f && (co.q0 > po.q0 * 2.0f || po.q0 > co.q0 * 2.0f))
                return false;
            const double zmax = std::max(co.z0, po.z0), zmin = std::min(co.z0, po.z0);
            return zmax - zmin <= 0.5 * zmax + 256.0;
        }

        void matchGroup(const uint32_t *cur, size_t curCount, const uint32_t *prev, size_t prevCount, bool sameData)
        {
            Frame &f = *m_cur;
            if (curCount == 1u && prevCount == 1u && sameData)
            {
                if (!sameDepthRange(f.objects[cur[0]], m_last->objects[prev[0]]))
                    return;
                m_prevTaken[prev[0]] = 1u;
                m_pairs.push_back({cur[0], prev[0], sameData});
                return;
            }
            m_candidates.clear();
            const float limit = m_maxMove * m_maxMove;
            for (size_t i = 0; i < curCount; ++i)
            {
                const Object &co = f.objects[cur[i]];
                for (size_t j = 0; j < prevCount; ++j)
                {
                    const Object &po = m_last->objects[prev[j]];
                    if (m_prevTaken[prev[j]] || (!sameData && po.primCount != co.primCount) || !sameDepthRange(co, po))
                        continue;
                    const float dx = co.cx() - po.cx(), dy = co.cy() - po.cy();
                    const float d = dx * dx + dy * dy;
                    if (d <= limit)
                        m_candidates.push_back({d, cur[i], prev[j]});
                }
            }
            // nearest first; things at the same place in the order they were drawn
            std::sort(m_candidates.begin(), m_candidates.end(), [](const Candidate &a, const Candidate &b)
                      { return a.distance != b.distance ? a.distance < b.distance : a.cur != b.cur ? a.cur < b.cur : a.prev < b.prev; });
            for (const Candidate &c : m_candidates)
            {
                if (m_curTaken[c.cur] || m_prevTaken[c.prev])
                    continue;
                m_curTaken[c.cur] = 1u;
                m_prevTaken[c.prev] = 1u;
                m_pairs.push_back({c.cur, c.prev, sameData});
            }
        }

        // Drawing a frame more than once has one trap: every pass after the first starts with
        // GS memory as the frame left it, not as it found it. A texture (or palette) that is
        // used first and replaced later in the same frame - the game recycles its texture memory
        // all the time - would be the new one already when the next pass uses it: the object gets
        // the wrong texture in every other picture and flickers. So, per image transfer:
        //  - if nothing in the rest of the frame reads what it writes (a texture loaded for the
        //    next frame), only the last pass does the transfer;
        //  - otherwise the pages it writes that were read earlier in the frame (texture,
        //    palette, copy source) are kept as they were before their first write and put back
        //    before each further pass.
        // A page the frame has drawn into before the transfer cannot be kept that way (its old
        // contents are gone by then) and is left as it is.
        // (SSX 3 also keeps small 8-bit textures in the alpha bytes of its display buffer; the
        // snow sparkle among them is replaced every few frames, at the end of the frame.)
        // The pages a run's texture reads. A big texture is often a whole buffer of which the
        // run uses a part (the 512 x 448 picture declared as 1024 x 512): for those, the part
        // its texture coordinates span.
        static void texturePages(const Frame &f, const Run &run, PageSet &set)
        {
            const GSDrawState &st = f.states[run.state];
            const GSContext &ctx = st.context;
            const uint32_t w = 1u << std::min<uint32_t>(ctx.tex0.tw, 10u), h = 1u << std::min<uint32_t>(ctx.tex0.th, 10u);
            uint32_t x0 = 0u, y0 = 0u, x1 = std::min(w, std::max<uint32_t>(ctx.tex0.tbw, 1u) * 64u) - 1u, y1 = h - 1u;
            if (w * h >= 256u * 256u)
            {
                float u0 = 1e30f, v0 = 1e30f, u1 = -1e30f, v1 = -1e30f;
                bool ok = true;
                const Prim *p = f.prims.data() + run.primStart;
                for (uint32_t k = 0; k < run.primCount && ok; ++k, ++p)
                    for (uint32_t i = 0; i < p->count; ++i)
                    {
                        const GSVertex &vtx = f.verts[p->vtx + i];
                        float u, v;
                        if (st.prim.fst)
                        {
                            u = static_cast<float>(vtx.u) * (1.0f / 16.0f);
                            v = static_cast<float>(vtx.v) * (1.0f / 16.0f);
                        }
                        else if (vtx.q != 0.0f)
                        {
                            u = vtx.s / vtx.q * static_cast<float>(w);
                            v = vtx.t / vtx.q * static_cast<float>(h);
                        }
                        else
                        {
                            ok = false;
                            break;
                        }
                        u0 = std::min(u0, u);
                        u1 = std::max(u1, u);
                        v0 = std::min(v0, v);
                        v1 = std::max(v1, v);
                    }
                // inside the texture (no wrapping), with a texel to spare for filtering
                if (ok && u0 >= -1.0f && v0 >= -1.0f && u1 <= static_cast<float>(w) + 1.0f && v1 <= static_cast<float>(h) + 1.0f)
                {
                    x0 = static_cast<uint32_t>(std::max(0.0f, std::floor(u0) - 1.0f));
                    y0 = static_cast<uint32_t>(std::max(0.0f, std::floor(v0) - 1.0f));
                    x1 = std::min(x1, static_cast<uint32_t>(std::ceil(u1) + 1.0f));
                    y1 = std::min(y1, static_cast<uint32_t>(std::ceil(v1) + 1.0f));
                }
            }
            addRectPages(set, ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, x0, y0, x1, y1);
            // mip levels 1..MXL (MIPTBP1: levels 1-3, MIPTBP2: levels 4-6)
            const uint32_t mxl = std::min<uint32_t>(static_cast<uint32_t>((ctx.tex1 >> 2) & 7u), 6u);
            for (uint32_t level = 1; level <= mxl; ++level)
            {
                const uint64_t reg = level <= 3u ? ctx.miptbp1 : ctx.miptbp2;
                const uint32_t shift = ((level - 1u) % 3u) * 20u;
                const uint32_t tbp = static_cast<uint32_t>((reg >> shift) & 0x3FFFu), tbw = static_cast<uint32_t>((reg >> (shift + 14u)) & 0x3Fu);
                addRectPages(set, tbp, tbw, ctx.tex0.psm, 0u, 0u, std::max(w >> level, 1u) - 1u, std::max(h >> level, 1u) - 1u);
            }
        }

        static inline bool indexedFormat(uint8_t psm)
        {
            return psm == GS_PSM_T8 || psm == GS_PSM_T4 || psm == GS_PSM_T8H || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
        }

        static void clutPages(const ClutLoad &c, PageSet &set)
        {
            if (c.tex0.csm == 0u)
            {
                // block-ordered at CBP: 16 x 16 (four blocks) for 8-bit textures, 8 x 2 (one) for 4-bit
                const bool small = c.tex0.psm == GS_PSM_T4 || c.tex0.psm == GS_PSM_T4HL || c.tex0.psm == GS_PSM_T4HH;
                const uint32_t last = std::min<uint32_t>(c.tex0.cbp + (small ? 0u : 3u), 0x3FFFu);
                set.set((c.tex0.cbp >> 5) & 511u);
                set.set((last >> 5) & 511u);
            }
            else
                addRectPages(set, c.tex0.cbp, c.texclut.cbw, c.tex0.cpsm, c.texclut.cou * 16u, c.texclut.cov, c.texclut.cou * 16u + 255u, c.texclut.cov);
        }

        // The pages an image transfer writes / reads (nothing for local->host and empty ones).
        static bool transferPages(const GSTransferCommand &t, PageSet &written, PageSet *read)
        {
            if (t.trxreg.rrw == 0u || t.trxreg.rrh == 0u || (t.direction != 0u && t.direction != 2u))
                return false;
            addRectPages(written, t.bitbltbuf.dbp, t.bitbltbuf.dbw, t.bitbltbuf.dpsm, t.trxpos.dsax, t.trxpos.dsay, t.trxpos.dsax + t.trxreg.rrw - 1u,
                         t.trxpos.dsay + t.trxreg.rrh - 1u);
            if (read && t.direction == 2u)
                addRectPages(*read, t.bitbltbuf.sbp, t.bitbltbuf.sbw, t.bitbltbuf.spsm, t.trxpos.ssax, t.trxpos.ssay, t.trxpos.ssax + t.trxreg.rrw - 1u,
                             t.trxpos.ssay + t.trxreg.rrh - 1u);
            return true;
        }

        void findHazards(Frame &f)
        {
            m_hazard.reset();
            m_drawn.reset();
            m_xferLate.assign(f.transfers.size(), 0u);
            m_statLateTransfers = 0;
            // PS2_FRAME_INTERP_HAZARD=all: every transfer in every pass, every page it writes put
            // back; =none: nothing put back, nothing left for the last pass (debug)
            static const char *s_mode = std::getenv("PS2_FRAME_INTERP_HAZARD");
            const bool all = s_mode && s_mode[0] == 'a', none = s_mode && s_mode[0] == 'n';

            // Backwards: is what a transfer writes read again before the frame ends?
            if (!all && !none)
            {
                PageSet readLater;
                for (size_t i = f.ops.size(); i-- > 0u;)
                {
                    const Op &op = f.ops[i];
                    if (op.kind == OpKind::Prims)
                    {
                        if (f.states[f.runs[op.a].state].prim.tme)
                            texturePages(f, f.runs[op.a], readLater);
                    }
                    else if (op.kind == OpKind::LoadClut)
                        clutPages(f.cluts[op.a], readLater);
                    else if (op.kind == OpKind::Transfer)
                    {
                        PageSet written;
                        if (!transferPages(f.transfers[op.a], written, nullptr))
                            continue;
                        if ((written & readLater).none())
                        {
                            m_xferLate[op.a] = 1u;
                            ++m_statLateTransfers;
                        }
                        PageSet unused;
                        transferPages(f.transfers[op.a], unused, &readLater); // (a copy reads its source)
                    }
                }
            }

            // Does the frame draw with the palette it found loaded (a paletted texture before its
            // first palette load)? Only then is that palette read, and put back between passes.
            m_needStartClut = false;
            if (m_haveStartClut)
                for (const Op &op : f.ops)
                {
                    if (op.kind == OpKind::LoadClut)
                    {
                        const GSTex0Reg &t = f.cluts[op.a].tex0;
                        if (indexedFormat(t.psm) && t.cld != 0u && t.cld < 6u)
                            break;
                    }
                    else if (op.kind == OpKind::Prims)
                    {
                        const GSDrawState &st = f.states[f.runs[op.a].state];
                        if (st.prim.tme && indexedFormat(st.context.tex0.psm))
                        {
                            m_needStartClut = true;
                            break;
                        }
                    }
                }

            // Forwards: pages read, then written by a transfer that every pass does.
            PageSet read;
            if (all)
                read.set();
            if (m_needStartClut)
                clutPages(m_startClut, read);
            for (const Op &op : f.ops)
            {
                if (op.kind == OpKind::Prims)
                {
                    const Run &run = f.runs[op.a];
                    const GSDrawState &st = f.states[run.state];
                    const GSContext &ctx = st.context;
                    if (st.prim.tme)
                        texturePages(f, run, read);
                    // what it draws into: the part of the buffers its vertices span (the scissor
                    // rectangle alone can be far larger than the buffer), depth only when written
                    const float ox = static_cast<float>(ctx.xyoffset.ofx >> 4), oy = static_cast<float>(ctx.xyoffset.ofy >> 4);
                    const float fx0 = std::max(static_cast<float>(ctx.scissor.x0), std::floor(run.x0 - ox));
                    const float fy0 = std::max(static_cast<float>(ctx.scissor.y0), std::floor(run.y0 - oy));
                    const float fx1 = std::min(static_cast<float>(ctx.scissor.x1), std::ceil(run.x1 - ox));
                    const float fy1 = std::min(static_cast<float>(ctx.scissor.y1), std::ceil(run.y1 - oy));
                    if (fx1 >= fx0 && fy1 >= fy0)
                    {
                        const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
                        const uint32_t x0 = static_cast<uint32_t>(fx0), y0 = static_cast<uint32_t>(fy0), x1 = static_cast<uint32_t>(fx1), y1 = static_cast<uint32_t>(fy1);
                        addRectPages(m_drawn, ctx.frame.fbp << 5, fbw, ctx.frame.psm, x0, y0, x1, y1);
                        if (!ctx.zbuf.zmask)
                            addRectPages(m_drawn, ctx.zbuf.zbp << 5, fbw, ctx.zbuf.psm, x0, y0, x1, y1);
                    }
                }
                else if (op.kind == OpKind::LoadClut)
                    clutPages(f.cluts[op.a], read);
                else if (op.kind == OpKind::Transfer)
                {
                    if (m_xferLate[op.a])
                        continue;
                    PageSet written;
                    if (!transferPages(f.transfers[op.a], written, &read))
                        continue;
                    const PageSet hazard = written & read;
                    m_hazard |= hazard & ~m_drawn;
                    if (m_hazardLog && hazard.any())
                    {
                        const GSTransferCommand &t = f.transfers[op.a];
                        std::fprintf(stderr, "[gs:interp] transfer (op %zu, dir %u) to %x (width %u, format %x) at %u,%u size %ux%u replaces what the frame read before: %zu page(s), %zu of them drawn into before (left alone)\n",
                                     static_cast<size_t>(&op - f.ops.data()), t.direction, t.bitbltbuf.dbp, t.bitbltbuf.dbw, t.bitbltbuf.dpsm, t.trxpos.dsax, t.trxpos.dsay, t.trxreg.rrw, t.trxreg.rrh,
                                     hazard.count(), (hazard & m_drawn).count());
                    }
                }
            }
            if (none)
                m_hazard.reset();
            m_undo.clear();
            m_undoPages.clear();
            m_saved.reset();
        }

        // First pass of several, before a transfer: the hazard pages it writes, as they are now
        // (not written yet in this frame), in the order a 64 x 32 32-bit image transfer of the
        // page sends them.
        void saveHazardPages(const GSTransferCommand &t)
        {
            PageSet written;
            if (m_hazard.none() || !m_vram || !transferPages(t, written, nullptr))
                return;
            const PageSet need = written & m_hazard & ~m_saved;
            if (need.none())
                return;
            // (data the renderer may still hold: a page that is, or was, part of a render target)
            if (m_ex)
                m_ex->SyncPages(need);
            else
                m_inner->Sync(GSSyncReason::DebugReadback);
            for (uint32_t page = 0; page < 512u; ++page)
            {
                if (!need.test(page) || (page + 1u) * kPageBytes > m_vramSize)
                    continue;
                const size_t at = m_undo.size();
                m_undo.resize(at + kPageBytes);
                uint32_t *out = reinterpret_cast<uint32_t *>(m_undo.data() + at);
                for (uint32_t y = 0; y < 32u; ++y)
                    for (uint32_t x = 0; x < 64u; ++x)
                        *out++ = GSMem::ReadCT32(m_vram, page * 32u, 1u, x, y);
                m_undoPages.push_back(static_cast<uint16_t>(page));
            }
            m_saved |= need;
        }

        void restoreHazardPages()
        {
            for (size_t i = 0; i < m_undoPages.size(); ++i)
            {
                GSTransferCommand cmd{};
                cmd.bitbltbuf.dbp = static_cast<uint32_t>(m_undoPages[i]) * 32u;
                cmd.bitbltbuf.dbw = 1u;
                cmd.bitbltbuf.dpsm = GS_PSM_CT32;
                cmd.trxreg.rrw = 64u;
                cmd.trxreg.rrh = 32u;
                cmd.direction = 0u;
                m_inner->BeginTransfer(cmd);
                m_inner->UploadImage(m_undo.data() + i * kPageBytes, kPageBytes);
            }
            // ... and the palette as the frame found it: the one the last frame loaded last
            // (a texture drawn before the frame loads a palette of its own uses that one)
            if (m_haveStartClut && m_needStartClut && !m_noClutRestore)
                m_inner->LoadClut(m_startClut.tex0, m_startClut.texclut);
        }

        // At the flip: which object of last frame is each object of this frame?
        //  1. same program and same input data (a model drawn again): by nearness when it is
        //     drawn several times (instances);
        //  2. what is left, same program and same number of primitives (animated models,
        //     particles: the data changes every frame): by nearness.
        void matchFrame()
        {
            const uint64_t tStart = nowNs();
            Frame &f = *m_cur;
            f.phaseNs[0] = f.phaseNs[1] = f.phaseNs[2] = f.phaseNs[3] = 0;
            ++m_matchCount;
            f.byKey.resize(f.objects.size());
            for (uint32_t i = 0; i < f.byKey.size(); ++i)
                f.byKey[i] = i;
            std::sort(f.byKey.begin(), f.byKey.end(), [&](uint32_t a, uint32_t b)
                      { return f.objects[a].key != f.objects[b].key ? f.objects[a].key < f.objects[b].key : a < b; });
            f.synth.clear();
            m_haveModel = false;
            if (m_last->objects.empty())
                return;
            m_pairs.clear();
            m_implausible.clear();
            m_curTaken.assign(f.objects.size(), 0u);
            m_prevTaken.assign(m_last->objects.size(), 0u);

            // 1. equal keys
            size_t i = 0, j = 0;
            const std::vector<uint32_t> &ck = f.byKey, &pk = m_last->byKey;
            while (i < ck.size() && j < pk.size())
            {
                const uint64_t a = f.objects[ck[i]].key, b = m_last->objects[pk[j]].key;
                if (a < b)
                    ++i;
                else if (b < a)
                    ++j;
                else
                {
                    size_t ie = i, je = j;
                    while (ie < ck.size() && f.objects[ck[ie]].key == a)
                        ++ie;
                    while (je < pk.size() && m_last->objects[pk[je]].key == a)
                        ++je;
                    const size_t before = m_pairs.size();
                    matchGroup(ck.data() + i, ie - i, pk.data() + j, je - j, true);
                    for (size_t k = before; k < m_pairs.size(); ++k)
                        m_curTaken[m_pairs[k].cur] = 1u;
                    i = ie;
                    j = je;
                }
            }

            // 2. the rest, by program and kind of drawing
            m_restCur.clear();
            m_restPrev.clear();
            for (uint32_t k = 0; k < f.objects.size(); ++k)
                if (!m_curTaken[k])
                    m_restCur.push_back(k);
            for (uint32_t k = 0; k < m_last->objects.size(); ++k)
                if (!m_prevTaken[k])
                    m_restPrev.push_back(k);
            auto byGroup = [](const std::vector<Object> &objects)
            {
                return [&objects](uint32_t a, uint32_t b)
                {
                    return objects[a].group != objects[b].group ? objects[a].group < objects[b].group : a < b;
                };
            };
            std::sort(m_restCur.begin(), m_restCur.end(), byGroup(f.objects));
            std::sort(m_restPrev.begin(), m_restPrev.end(), byGroup(m_last->objects));
            i = j = 0;
            while (i < m_restCur.size() && j < m_restPrev.size())
            {
                const uint64_t a = f.objects[m_restCur[i]].group, b = m_last->objects[m_restPrev[j]].group;
                if (a < b)
                    ++i;
                else if (b < a)
                    ++j;
                else
                {
                    size_t ie = i, je = j;
                    while (ie < m_restCur.size() && f.objects[m_restCur[ie]].group == a)
                        ++ie;
                    while (je < m_restPrev.size() && m_last->objects[m_restPrev[je]].group == a)
                        ++je;
                    matchGroup(m_restCur.data() + i, ie - i, m_restPrev.data() + j, je - j, false);
                    i = ie;
                    j = je;
                }
            }

            const uint64_t t0 = nowNs();
            for (const Pair &pair : m_pairs)
            {
                matchPrims(f.objects[pair.cur], m_last->objects[pair.prev], pair.sameData);
                checkObject(f.objects[pair.cur]);
            }
            const uint64_t tm = nowNs();
            fitTextureModels();
            if (static_cast<size_t>(f.matched) * 2u < f.prims.size())
            {
                // a new scene (the flip shows this frame once): nothing more to work out
                f.phaseNs[0] = t0 - tStart;
                f.phaseNs[1] = tm - t0;
                return;
            }
            fitCamera();
            static const bool s_time = std::getenv("PS2_FRAME_INTERP_DEBUG") != nullptr;
            const uint64_t tc = s_time ? nowNs() : 0u;
            classifyObjects();
            settleMovers();
            if (s_time)
                std::fprintf(stderr, "[gs:interp] camera fit %.2f ms (%zu samples), classify %.2f ms\n", (tc - tm) / 1e6, m_samples.size(), (nowNs() - tc) / 1e6);
            const uint64_t t1 = nowNs();
            f.phaseNs[3] = t1 - tm;
            if (const char *dump = std::getenv("PS2_FRAME_INTERP_DUMPMATCH"))
            {
                // every vertex of the frame with its match (debug: for studying the motion offline)
                static FILE *out = std::fopen(dump, "wb");
                if (out)
                {
                    for (const Object &o : f.objects)
                        for (uint32_t k = 0; k < o.primCount; ++k)
                        {
                            const Prim &pr = f.prims[o.primStart + k];
                            for (uint32_t v = 0; v < pr.count; ++v)
                            {
                                const GSVertex &c = f.verts[pr.vtx + v];
                                const bool has = hasPrev(pr.prev);
                                const GSVertex &q = has ? m_last->verts[pr.prev + v] : c;
                                const float rec[12] = {static_cast<float>(m_matchCount), static_cast<float>(static_cast<uint32_t>(o.key)), has ? 1.0f : 0.0f, static_cast<float>(&o - f.objects.data()),
                                                       c.x, c.y, static_cast<float>(c.z), c.q, q.x, q.y, static_cast<float>(q.z), q.q};
                                std::fwrite(rec, sizeof(float), 12, out);
                            }
                        }
                    std::fflush(out);
                }
            }
            recoverUnmatched();
            pairTextureModels();
            f.haveModel = m_haveModel;
            f.modelShare = m_modelShare;
            f.phaseNs[0] = t0 - tStart;
            f.phaseNs[1] = tm - t0;
            f.phaseNs[2] = nowNs() - t1;
            if (const char *hold = std::getenv("PS2_FRAME_INTERP_HOLD"))
            {
                // PS2_FRAME_INTERP_HOLD=first,last,match: those objects keep this frame's positions
                // in the in-between pictures of that match (debug: which object is it?)
                unsigned first = 0, last = 0;
                unsigned long long match = 0;
                if (std::sscanf(hold, "%u,%u,%llu", &first, &last, &match) == 3 && match == m_matchCount)
                    for (size_t k = first; k <= last && k < f.objects.size(); ++k)
                        for (uint32_t q = 0; q < f.objects[k].primCount; ++q)
                            f.prims[f.objects[k].primStart + q].prev = kRejected;
            }
            if (const char *rect = std::getenv("PS2_FRAME_INTERP_PRIMRECT"))
            {
                // PS2_FRAME_INTERP_PRIMRECT=x0,y0,x1,y1,match (screen pixels): the primitives of
                // that match that touch the rectangle, with where they are taken from (debug)
                float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
                unsigned long long match = 0;
                if (std::sscanf(rect, "%f,%f,%f,%f,%llu", &x0, &y0, &x1, &y1, &match) == 5 && match == m_matchCount)
                {
                    size_t objectIndex = 0;
                    for (const Run &run : f.runs)
                    {
                        const GSContext &ctx = f.states[run.state].context;
                        const float ox = static_cast<float>(ctx.xyoffset.ofx >> 4), oy = static_cast<float>(ctx.xyoffset.ofy >> 4);
                        for (uint32_t k = 0; k < run.primCount; ++k)
                        {
                            const uint32_t index = run.primStart + k;
                            const Prim &pr = f.prims[index];
                            float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
                            for (uint32_t v = 0; v < pr.count; ++v)
                            {
                                bx0 = std::min(bx0, f.pos[pr.vtx + v].x - ox);
                                bx1 = std::max(bx1, f.pos[pr.vtx + v].x - ox);
                                by0 = std::min(by0, f.pos[pr.vtx + v].y - oy);
                                by1 = std::max(by1, f.pos[pr.vtx + v].y - oy);
                            }
                            if (bx1 < x0 || bx0 > x1 || by1 < y0 || by0 > y1)
                                continue;
                            while (objectIndex + 1u < f.objects.size() && index >= f.objects[objectIndex].primStart + f.objects[objectIndex].primCount)
                                ++objectIndex;
                            const Object &o = f.objects[objectIndex];
                            std::fprintf(stderr, "[gs:interp] prim %u object %zu prog %u fbp %x tex %x %s motion %u%s%s:", index, objectIndex, o.prog, ctx.frame.fbp, static_cast<unsigned>(ctx.tex0.tbp0),
                                         pr.prev == kRejected ? "held" : pr.prev == kNone ? "alone" : (pr.prev & kSynth) ? "placed" : "matched", o.motion, o.clipped ? " clipped" : "", o.rejected ? " object-held" : "");
                            for (uint32_t v = 0; v < pr.count; ++v)
                            {
                                const GSVertex &c = f.verts[pr.vtx + v];
                                std::fprintf(stderr, " (%.3f,%.3f q%.9g z%.0f st %.3f,%.3f", c.x - ox, c.y - oy, c.q, static_cast<double>(c.z), c.s / c.q, c.t / c.q);
                                if (hasPrev(pr.prev))
                                {
                                    const GSVertex &q = (pr.prev & kSynth) ? f.synth[(pr.prev & ~kSynth) + v] : m_last->verts[pr.prev + v];
                                    std::fprintf(stderr, " <- %.3f,%.3f q%.9g z%.0f st %.3f,%.3f", q.x - ox, q.y - oy, q.q, static_cast<double>(q.z), q.s / q.q, q.t / q.q);
                                }
                                float mx = 0.0f, my = 0.0f, mq = 0.0f;
                                if (predict(c, mx, my, mq))
                                    std::fprintf(stderr, " ~ %.3f,%.3f q%.9g", mx - ox, my - oy, mq);
                                std::fprintf(stderr, ")");
                            }
                            std::fprintf(stderr, "\n");
                        }
                    }
                }
            }
            if (const char *rect = std::getenv("PS2_FRAME_INTERP_OBJRECT"))
            {
                // PS2_FRAME_INTERP_OBJRECT=x0,y0,x1,y1 (GS coordinates): the objects whose centre
                // lies in that rectangle, and what became of their primitives (debug)
                float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
                if (std::sscanf(rect, "%f,%f,%f,%f", &x0, &y0, &x1, &y1) == 4)
                {
                    std::vector<int32_t> pairOf(f.objects.size(), -1);
                    for (size_t k = 0; k < m_pairs.size(); ++k)
                        pairOf[m_pairs[k].cur] = static_cast<int32_t>(k);
                    for (size_t k = 0; k < f.objects.size(); ++k)
                    {
                        const Object &o = f.objects[k];
                        if (o.vertices == 0u || o.cx() < x0 || o.cx() > x1 || o.cy() < y0 || o.cy() > y1)
                            continue;
                        uint32_t real = 0, synth = 0, rejected = 0, none = 0;
                        float maxMove = 0.0f;
                        for (uint32_t q = 0; q < o.primCount; ++q)
                        {
                            const Prim &pr = f.prims[o.primStart + q];
                            if (pr.prev == kRejected)
                                ++rejected;
                            else if (pr.prev == kNone)
                                ++none;
                            else if (pr.prev & kSynth)
                                ++synth;
                            else
                            {
                                ++real;
                                for (uint32_t v = 0; v < pr.count; ++v)
                                    maxMove = std::max(maxMove, std::max(std::fabs(f.pos[pr.vtx + v].x - m_last->pos[pr.prev + v].x), std::fabs(f.pos[pr.vtx + v].y - m_last->pos[pr.prev + v].y)));
                            }
                        }
                        std::fprintf(stderr, "[gs:interp] match %llu object %zu key %llx prog %u at %.0f,%.0f: %u prims = %u matched (largest move %.0f) + %u placed + %u rejected + %u alone; %s, motion %u%s (%.1f,%.1f) q0 %g\n",
                                     static_cast<unsigned long long>(m_matchCount), k, static_cast<unsigned long long>(o.key), o.prog, o.cx(), o.cy(), o.primCount, real, maxMove, synth, rejected, none,
                                     pairOf[k] < 0 ? "no partner" : m_pairs[pairOf[k]].sameData ? "same data" : "by kind", o.motion, o.hasMotion ? "" : " (no motion)", o.dx, o.dy, o.q0);
                    }
                }
            }
            static const bool s_dbg = std::getenv("PS2_FRAME_INTERP_DEBUG") != nullptr;
            if (s_dbg)
            {
                // per VU1 program: objects, objects paired, primitives, primitives left alone
                struct Sum
                {
                    uint32_t objects = 0, paired = 0, prims = 0, alone = 0;
                };
                std::unordered_map<uint32_t, Sum> sums;
                std::vector<uint8_t> paired(f.objects.size(), 0u);
                for (const Pair &pair : m_pairs)
                    paired[pair.cur] = 1u;
                for (size_t k = 0; k < f.objects.size(); ++k)
                {
                    const Object &o = f.objects[k];
                    Sum &sum = sums[static_cast<uint32_t>(o.key)];
                    ++sum.objects;
                    sum.paired += paired[k];
                    sum.prims += o.primCount;
                    for (uint32_t q = 0; q < o.primCount; ++q)
                        sum.alone += hasPrev(f.prims[o.primStart + q].prev) ? 0u : 1u;
                }
                std::string line;
                for (const auto &[pc, sum] : sums)
                {
                    if (sum.alone == 0u)
                        continue;
                    char item[96];
                    std::snprintf(item, sizeof(item), " %x: %u/%u prims alone (%u/%u objects paired)", pc, sum.alone, sum.prims, sum.paired, sum.objects);
                    line += item;
                }
                std::fprintf(stderr, "[gs:interp] left alone by program:%s\n", line.c_str());
            }
        }

        // Execute the buffered ops from `from`. t < 1: an in-between picture (matched vertices
        // part of the way from last frame's positions; nothing that leaves the GS).
        // `save`: the first of several passes (keeps the hazard pages as it goes, see findHazards).
        void replay(Frame &f, const Frame *prev, size_t from, float t, bool save = false)
        {
            const bool between = t < 1.0f && prev != nullptr;
            GSPrimitiveBatch batch;
            bool skipUpload = false;
            for (size_t i = from; i < f.ops.size(); ++i)
            {
                const Op &op = f.ops[i];
                switch (op.kind)
                {
                case OpKind::Prims:
                {
                    const Run &run = f.runs[op.a];
                    batch.state = f.states[run.state];
                    if (between && m_dbgSkip != 0u)
                    {
                        const uint32_t fbp = batch.state.context.frame.fbp;
                        if ((m_dbgSkip & 2u) || ((m_dbgSkip & 1u) && fbp != 0u && fbp != 0x70u) || ((m_dbgSkip & 16u) && fbp == 0x70u) || ((m_dbgSkip & 32u) && fbp == 0u))
                            break;
                    }
                    const Prim *p = f.prims.data() + run.primStart;
                    for (uint32_t k = 0; k < run.primCount; ++k, ++p)
                    {
                        const GSVertex *cv = f.verts.data() + p->vtx;
                        batch.vertexCount = p->count;
                        if (between && hasPrev(p->prev) && (m_dbgOnly == 0u || p->count == m_dbgOnly))
                        {
                            const GSVertex *pv = (p->prev & kSynth) ? f.synth.data() + (p->prev & ~kSynth) : prev->verts.data() + p->prev;
                            const UvModel *rule = p->uv != 0u && p->uv <= f.uvModels.size() && f.uvModels[p->uv - 1u].hasPrev ? &f.uvModels[p->uv - 1u] : nullptr;
                            for (uint32_t v = 0; v < p->count; ++v)
                            {
                                GSVertex &o = batch.vertices[v];
                                const GSVertex &a = pv[v];
                                const GSVertex &b = cv[v];
                                // Move the vertex in space, not on the screen: with w = 1 / q the
                                // position is averaged with the weights (1 - t) wa and t wb, which is
                                // what drawing with a camera half-way between the two frames gives.
                                // Flat things stay flat, layers and decals stay on their surface, and
                                // textures do not swim on triangles that come close to the camera.
                                // Without perspective (q the same in both) it is the plain average.
                                // (Written with the ratio of the two q, so that layers of one
                                // surface - same positions, same ratio - come out bit for bit the
                                // same, depth included, whatever their own q is.)
                                if (a.q > 0.0f && b.q > 0.0f && a.q < b.q * 8.0f && b.q < a.q * 8.0f)
                                {
                                    const float ka = (1.0f - t) / (a.q / b.q), kb = t;
                                    const float inv = 1.0f / (ka + kb);
                                    o.x = (a.x * ka + b.x * kb) * inv;
                                    o.y = (a.y * ka + b.y * kb) * inv;
                                    o.z = (a.z * static_cast<double>(ka) + b.z * static_cast<double>(kb)) * static_cast<double>(inv);
                                    o.q = b.q * inv;
                                }
                                else
                                {
                                    o.x = a.x + (b.x - a.x) * t;
                                    o.y = a.y + (b.y - a.y) * t;
                                    o.z = a.z + (b.z - a.z) * static_cast<double>(t);
                                    o.q = b.q;
                                }
                                // Only the position moves. Colour, fog and texture coordinates stay
                                // this frame's: they may scroll, wrap or pulse from frame to frame.
                                // (Except a texture laid over the scene by a rule: half-way between
                                // what last frame's rule gave where the vertex was and this frame's.)
                                const float ratio = b.q != 0.0f ? o.q / b.q : 1.0f;
                                if (rule && perspective(a.q) && perspective(b.q))
                                {
                                    double basis[4];
                                    uvBasis(a, basis);
                                    const double s0 = rule->lPrev[0][0] * basis[0] + rule->lPrev[0][1] * basis[1] + rule->lPrev[0][2] * basis[2] + rule->lPrev[0][3];
                                    const double t0 = rule->lPrev[1][0] * basis[0] + rule->lPrev[1][1] * basis[1] + rule->lPrev[1][2] * basis[2] + rule->lPrev[1][3];
                                    const double s1 = static_cast<double>(b.s) / b.q, t1 = static_cast<double>(b.t) / b.q;
                                    o.s = static_cast<float>((s0 + (s1 - s0) * t) * o.q);
                                    o.t = static_cast<float>((t0 + (t1 - t0) * t) * o.q);
                                }
                                else
                                {
                                    o.s = b.s * ratio;
                                    o.t = b.t * ratio;
                                }
                                o.r = b.r;
                                o.g = b.g;
                                o.b = b.b;
                                o.a = b.a;
                                o.u = b.u;
                                o.v = b.v;
                                o.fog = b.fog;
                            }
                        }
                        else
                        {
                            for (uint32_t v = 0; v < p->count; ++v)
                                batch.vertices[v] = cv[v];
                        }
                        if (between && m_probe && batch.vertexCount < 3u)
                        {
                            // sprites, lines and points near the probe
                            float x0 = cv[0].x, x1 = cv[0].x, y0 = cv[0].y, y1 = cv[0].y;
                            for (uint32_t v = 1; v < p->count; ++v)
                            {
                                x0 = std::min(x0, cv[v].x);
                                x1 = std::max(x1, cv[v].x);
                                y0 = std::min(y0, cv[v].y);
                                y1 = std::max(y1, cv[v].y);
                            }
                            if (m_probeX >= x0 - 1.0f && m_probeX <= x1 + 1.0f && m_probeY >= y0 - 1.0f && m_probeY <= y1 + 1.0f)
                            {
                                uint64_t key = 0;
                                const uint32_t primIndex = static_cast<uint32_t>(p - f.prims.data());
                                for (const Object &o : f.objects)
                                    if (primIndex >= o.primStart && primIndex < o.primStart + o.primCount)
                                        key = o.key;
                                const GSContext &c = batch.state.context;
                                std::fprintf(stderr, "[gs:interp] probe: obj %llx %u-vertex prim type %u %s tbp %x psm %x cbp %x tme %d abe %d alpha %llx test %llx fbp %x | cur (%.2f,%.2f)(%.2f,%.2f) z %.0f rgba %u,%u,%u,%u\n",
                                             static_cast<unsigned long long>(key), p->count, static_cast<unsigned>(batch.state.prim.type),
                                             p->prev == kNone ? "alone" : p->prev == kRejected ? "rejected" : (p->prev & kSynth) ? "placed" : "matched",
                                             c.tex0.tbp0, c.tex0.psm, c.tex0.cbp, batch.state.prim.tme ? 1 : 0, batch.state.prim.abe ? 1 : 0,
                                             static_cast<unsigned long long>(c.alpha), static_cast<unsigned long long>(c.test), c.frame.fbp,
                                             cv[0].x, cv[0].y, cv[p->count - 1u].x, cv[p->count - 1u].y, cv[0].z, cv[0].r, cv[0].g, cv[0].b, cv[0].a);
                            }
                        }
                        if (between && m_probe && batch.vertexCount == 3u)
                        {
                            // PS2_FRAME_INTERP_PROBE=x,y (GS coordinates): what is drawn over that point
                            static const bool s_probeCur = std::getenv("PS2_FRAME_INTERP_PROBECUR") != nullptr;
                            const GSVertex *v = s_probeCur ? cv : batch.vertices.data(); // where it is this frame / in the in-between picture
                            auto side = [&](const GSVertex &a, const GSVertex &b)
                            { return (b.x - a.x) * (m_probeY - a.y) - (b.y - a.y) * (m_probeX - a.x); };
                            const float d0 = side(v[0], v[1]), d1 = side(v[1], v[2]), d2 = side(v[2], v[0]);
                            if (std::fabs(d0 + d1 + d2) > 1.0f && ((d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0)))
                            {
                                uint64_t key = 0;
                                uint32_t objIndex = 0, objPrims = 0;
                                const uint32_t primIndex = static_cast<uint32_t>(p - f.prims.data());
                                for (const Object &o : f.objects)
                                    if (primIndex >= o.primStart && primIndex < o.primStart + o.primCount)
                                    {
                                        key = o.key;
                                        objIndex = primIndex - o.primStart;
                                        objPrims = o.primCount;
                                    }
                                const GSVertex *pv = !hasPrev(p->prev) ? nullptr : (p->prev & kSynth) ? f.synth.data() + (p->prev & ~kSynth) : prev->verts.data() + p->prev;
                                std::fprintf(stderr, "[gs:interp] probe: obj %llx prim %u/%u kick %u %s tbp %x abe %d | cur (%.1f,%.1f)(%.1f,%.1f)(%.1f,%.1f) st (%.3f,%.3f)(%.3f,%.3f)(%.3f,%.3f) z %.0f q %g %g %g",
                                             static_cast<unsigned long long>(key), objIndex, objPrims, p->kick,
                                             p->prev == kNone ? "alone" : p->prev == kRejected ? "rejected" : (p->prev & kSynth) ? "placed" : "matched",
                                             batch.state.context.tex0.tbp0, batch.state.prim.abe ? 1 : 0, cv[0].x, cv[0].y, cv[1].x, cv[1].y, cv[2].x, cv[2].y,
                                             cv[0].s / cv[0].q, cv[0].t / cv[0].q, cv[1].s / cv[1].q, cv[1].t / cv[1].q, cv[2].s / cv[2].q, cv[2].t / cv[2].q, cv[0].z, cv[0].q, cv[1].q, cv[2].q);
                                if (pv)
                                    std::fprintf(stderr, " | prev q %g %g %g (%.1f,%.1f)(%.1f,%.1f)(%.1f,%.1f) st (%.3f,%.3f)(%.3f,%.3f)(%.3f,%.3f) z %.0f", pv[0].q, pv[1].q, pv[2].q, pv[0].x, pv[0].y, pv[1].x, pv[1].y, pv[2].x, pv[2].y,
                                                 pv[0].s / pv[0].q, pv[0].t / pv[0].q, pv[1].s / pv[1].q, pv[1].t / pv[1].q, pv[2].s / pv[2].q, pv[2].t / pv[2].q, pv[0].z);
                                std::fprintf(stderr, "\n");
                            }
                        }
                        m_inner->Submit(batch);
                    }
                    break;
                }
                case OpKind::LoadClut:
                    if (between && (m_dbgSkip & 8u))
                        break;
                    m_inner->LoadClut(f.cluts[op.a].tex0, f.cluts[op.a].texclut);
                    break;
                case OpKind::Transfer:
                    // A local->host transfer only feeds a read-back: real frame only.
                    // One the rest of the frame does not use: last pass only.
                    skipUpload = between && (f.transfers[op.a].direction == 1u || (m_dbgSkip & 4u) || (op.a < m_xferLate.size() && m_xferLate[op.a]));
                    if (!skipUpload && f.transfers[op.a].direction == 1u && m_asyncReads)
                    {
                        // A read-back whose data is wanted later (the next thing recorded is
                        // its deferred read): the renderer reads it without waiting, if it can.
                        size_t j = i + 1u;
                        while (j < f.ops.size() && (f.ops[j].kind == OpKind::Flush || f.ops[j].kind == OpKind::TextureFlush))
                            ++j;
                        if (j < f.ops.size() && f.ops[j].kind == OpKind::Readback)
                        {
                            Readback &rb = f.readbacks[f.ops[j].a];
                            if (rb.done && !rb.handled && innerReadbackAsync(f.transfers[op.a], sizedReadback(rb.done, rb.bytes)))
                            {
                                rb.handled = true;
                                rb.done = nullptr;
                                break;
                            }
                        }
                    }
                    if (!skipUpload)
                    {
                        if (save)
                            saveHazardPages(f.transfers[op.a]);
                        m_inner->BeginTransfer(f.transfers[op.a]);
                    }
                    break;
                case OpKind::Upload:
                    if (!skipUpload)
                        m_inner->UploadImage(f.bytes.data() + op.a, op.b);
                    break;
                case OpKind::Flush:
                    m_inner->Flush();
                    break;
                case OpKind::TextureFlush:
                    m_inner->TextureFlush();
                    break;
                case OpKind::Readback:
                    if (!between && !f.readbacks[op.a].handled)
                    {
                        Readback &rb = f.readbacks[op.a];
                        std::vector<uint8_t> data(rb.bytes, 0u);
                        const uint32_t got = m_inner->ConsumeLocalToHostBytes(data.data(), rb.bytes);
                        if (rb.done)
                            rb.done(std::move(data), got);
                        rb.done = nullptr;
                    }
                    break;
                }
            }
        }

        // Something needs the renderer's state now: draw what is buffered as it is.
        void drain()
        {
            flushHeld();
            waitRenderIdle(); // the frames handed over come first
            Frame &f = *m_cur;
            if (f.doneOps >= f.ops.size())
                return;
            bool drawing = false;
            for (size_t i = f.doneOps; i < f.ops.size(); ++i)
                drawing = drawing || (f.ops[i].kind != OpKind::Flush && f.ops[i].kind != OpKind::TextureFlush);
            replay(f, nullptr, f.doneOps, 1.0f);
            f.doneOps = f.ops.size();
            m_runOpen = false;
            if (drawing)
            {
                f.broken = true;
                ++m_statDrains;
            }
        }

        // -------------------------------------------------------------------------------------
        // The flip, in two halves. On the GS thread (flip): the finished frame is matched with
        // the last one and handed over. On the render thread (runJob; with
        // PS2_FRAME_INTERP_THREAD=0 right there on the GS thread): the passes. So the GS thread
        // takes in the next frame while this one is being drawn two or more times.
        // What belongs to which half: the frame being recorded (m_cur), the last one (m_last)
        // and everything the matching uses are the GS thread's; a frame handed over is only
        // read by the matching of the next (never changed), while the render thread draws it;
        // the schedule, the texture bookkeeping and the statistics are the render thread's.
        // -------------------------------------------------------------------------------------
        struct Job
        {
            std::shared_ptr<Frame> frame, prev;
            GSPresentationRequest request{};
            uint64_t flipNs = 0, issuedNs = 0;
            uint64_t readyNs = 0; // handed to the render thread (which may still be busy)
            uint32_t factor = 1;
            bool canInterp = false; // matched well enough against a whole previous frame
            bool cut = false;       // a new scene
            bool plain = false;     // nothing to match: drawn and shown once
        };

        // A fresh frame to record into (its vectors keep their size from earlier use).
        std::shared_ptr<Frame> acquireFrame()
        {
            Frame *frame = nullptr;
            {
                std::lock_guard<std::mutex> lock(m_poolMutex);
                if (!m_pool.empty())
                {
                    frame = m_pool.back().release();
                    m_pool.pop_back();
                }
            }
            if (!frame)
                frame = new Frame();
            return std::shared_ptr<Frame>(frame, [this](Frame *done)
                                          {
                                              done->clear();
                                              std::lock_guard<std::mutex> lock(m_poolMutex);
                                              m_pool.emplace_back(done);
                                          });
        }

        void renderMain()
        {
            ps2NameThisThread("GS render thread");
            std::unique_lock<std::mutex> lock(m_jobMutex);
            for (;;)
            {
                m_jobCv.wait(lock, [this]
                             { return m_jobQueued || m_quit; });
                if (!m_jobQueued)
                    return; // (quit, nothing left)
                Job job = std::move(m_job);
                m_job = Job{};
                m_jobQueued = false;
                m_jobRunning = true;
                lock.unlock();
                m_jobCv.notify_all(); // room for the next
                runJob(job);
                job = Job{}; // the frames, before saying so
                lock.lock();
                m_jobRunning = false;
                m_jobCv.notify_all();
            }
        }

        // Hands a frame to the render thread; waits while the one before is still waiting
        // (so the GS thread is never more than one frame ahead of the one being drawn).
        void submitJob(Job &&job)
        {
            const uint64_t readyNs = job.readyNs = nowNs();
            if (!m_threaded)
            {
                runJob(job);
                return;
            }
            {
                std::unique_lock<std::mutex> lock(m_jobMutex);
                m_jobCv.wait(lock, [this]
                             { return !m_jobQueued; });
                m_job = std::move(job);
                m_jobQueued = true;
            }
            m_jobCv.notify_all();
            const uint64_t waited = nowNs() - readyNs;
            if (waited > kSlowNs)
                ps2SlowLog("GS thread", "waited for the render thread to take the frame", waited);
        }

        void waitRenderIdle()
        {
            if (!m_threaded)
                return;
            std::unique_lock<std::mutex> lock(m_jobMutex);
            m_jobCv.wait(lock, [this]
                         { return !m_jobQueued && !m_jobRunning; });
        }

        void flip(const GSPresentationRequest &request)
        {
            flushHeld();
            if (!m_active)
            {
                s_nextDueNs.store(0u, std::memory_order_relaxed);
                m_inner->QueuePresentSnapshot(request);
                if (g_gsInterpPassHook)
                    g_gsInterpPassHook(m_inner.get(), 1u, 1u, 1.0f);
                m_active = s_wantFactor.load(std::memory_order_relaxed) > 1u;
                if (m_active)
                {
                    // switched on: record from here
                    waitRenderIdle();
                    m_cur = acquireFrame();
                    m_cur->complete = true;
                    m_last = acquireFrame();
                    m_runOpen = false;
                    m_transferBegun = false;
                    m_haveSchedule = false;
                    m_haveStartClut = false; // not recorded: unknown
                }
                return;
            }
            closeObject();
            Frame &f = *m_cur;
            Job job;
            job.frame = m_cur;
            job.request = request;
            job.flipNs = nowNs();
            job.issuedNs = g_ps2FlipIssuedNs.exchange(0u, std::memory_order_relaxed);
            job.factor = std::clamp<uint32_t>(s_wantFactor.load(std::memory_order_relaxed), 1u, kMaxFactor);
            const uint32_t factor = job.factor;
            // Test mode (PS2_FRAME_INTERP_TEST=1): every other game frame is only drawn, not
            // remembered, so the in-between picture of the next frame (made from the frames on
            // either side) can be compared with it.
            static const bool s_skipTest = std::getenv("PS2_FRAME_INTERP_TEST") != nullptr;
            if (f.prims.empty() || (s_skipTest && (++m_testCount & 1u) == 0u))
            {
                // (Also a flip with nothing drawn since the last one - a second display register,
                // a paused game: shown as it is, the last drawn frame stays "last frame".)
                job.plain = true;
            }
            else
            {
                matchFrame();
                f.matchNs = nowNs() - job.flipNs;
                // In-between pictures need: the whole frame still buffered, a previous frame to
                // move from, and most of the matches believable (a camera cut matches the same
                // models at unrelated places; also when less than half of the frame found a
                // partner: a new scene).
                const uint32_t keyed = f.matched + f.rejected;
                job.cut = (keyed != 0u && f.rejected * 4u > keyed) || static_cast<size_t>(f.matched) * 2u < f.prims.size();
                job.canInterp = f.complete && !f.broken && m_last->complete && factor > 1u && f.matched != 0u && !job.cut;
                job.prev = m_last;
                f.complete = true;
                m_last = m_cur; // this frame becomes "last frame" (its positions and objects)
            }
            submitJob(std::move(job));
            m_cur = acquireFrame();
            m_runOpen = false;
            m_transferBegun = false;
            m_active = factor > 1u;
            m_cur->complete = m_active;
            if (!m_active)
            {
                // switched off: what was handed over first, then straight to the renderer
                waitRenderIdle();
                m_last = acquireFrame();
            }
        }

        // The passes of one frame (render thread).
        void runJob(Job &job)
        {
            Frame &f = *job.frame;
            const Frame *prev = job.prev.get();
            const uint32_t factor = job.factor;
            ++m_flipCount;
            if (job.plain)
            {
                replay(f, nullptr, f.doneOps, 1.0f);
                s_nextDueNs.store(m_haveSchedule ? m_lastShowReal + 1u : 0u, std::memory_order_relaxed);
                m_inner->QueuePresentSnapshot(job.request);
                if (g_gsInterpPassHook)
                    g_gsInterpPassHook(m_inner.get(), 1u, 1u, 1.0f);
                noteLastClut(f);
                return;
            }
            m_statMatchNs += f.matchNs;
            const bool cut = job.cut;
            // Not keeping up - this frame waited more than half a frame period for the render
            // thread to finish the one before: no extra pictures for a while, so that the game's
            // own rate is kept. Half a second at first; when it happens again soon after the extra
            // pictures are back, twice as long each time (up to 8 s): switching between 60 and
            // 120 pictures a second all the time looks worse than staying at 60 in a heavy scene.
            // (How long ago the game issued the flip does not count: in a heavy scene most of
            // that is the VU1 and GS work before the frame gets here, which fewer pictures do
            // not shorten. Only far behind - 3 periods - it does.)
            const uint64_t startNs = nowNs();
            const uint64_t lag = job.issuedNs != 0u && job.issuedNs <= startNs ? startNs - job.issuedNs : 0u;
            const uint64_t queued = m_threaded && job.readyNs != 0u && job.readyNs <= startNs ? startNs - job.readyNs : 0u;
            if (m_shedding && factor > 1u && (queued > m_periodNs / 2u || lag > 3u * m_periodNs))
            {
                if (m_flipCount >= m_shedUntil)
                {
                    ++m_statSheds;
                    m_shedFrames = m_flipCount < m_shedUntil + 90u ? std::min<uint64_t>(m_shedFrames * 2u, 480u) : 30u;
                }
                m_shedUntil = m_flipCount + m_shedFrames;
            }
            const bool shed = m_flipCount < m_shedUntil;
            m_statLagNs += lag;
            m_statQueuedNs += queued;
            const bool usable = job.canInterp && !shed;
            const uint32_t passes = usable ? factor : 1u;
            static const bool s_log = std::getenv("PS2_FRAME_INTERP_LOG") != nullptr;
            if (s_log)
                std::fprintf(stderr, "[gs:interp] flip (match %.2f ms = pair %.2f + prims %.2f + camera %.2f + place %.2f): %zu prims in %zu objects, %zu ops, matched %u (+%u placed exactly, +%u carried along) rejected %u, camera model %s (%.0f%% of matched samples), done ops %zu%s -> %u pass(es)%s\n",
                             f.matchNs / 1e6, f.phaseNs[0] / 1e6, f.phaseNs[1] / 1e6, f.phaseNs[3] / 1e6, f.phaseNs[2] / 1e6, f.prims.size(), f.objects.size(), f.ops.size(), f.matched, f.recovered, f.carried, f.rejected, f.haveModel ? "yes" : "no", f.haveModel ? 100.0f * f.modelShare : 0.0f, f.doneOps, f.broken ? " (broken)" : "", passes, cut ? " (cut)" : "");

            // When to show them. The pictures follow a steady schedule: the real one every game
            // frame period, the in-between ones at even steps before it. The schedule is only as
            // early as the pictures can be ready: when the first picture of a frame is done later
            // than its time (see below), the schedule moves back at once; when there is slack,
            // it creeps forward. So the delay settles a little above the longest the renderer
            // took lately, and the pictures come at even intervals instead of late.
            // How fast it creeps forward decides how even: creeping up to the edge again within
            // a quarter of a second, it is pushed back three times a second in an ordinary race
            // (by 1.4 ms on average - a sixth of the time between two pictures).
            // PS2_FRAME_INTERP_STEADY=1 holds what a push added (m_holdNs) and gives it back
            // slowly, over some ten seconds. Off by default: in the one run with it on, the
            // schedule was pushed less than half as often, but the render thread fell behind
            // three times as often (22-30 ms for a frame, most of it waiting for the GPU) and
            // the in-between pictures were dropped for about a third of the race. Why the two
            // go together is not understood yet.
            const uint64_t period = updatePeriod(job.flipNs);
            const uint64_t step = period / factor;
            uint64_t showReal = job.flipNs + step * (factor - 1u) + 8000000ull; // a first guess
            if (m_haveSchedule)
            {
                const uint64_t predicted = m_lastShowReal + period;
                const int64_t off = static_cast<int64_t>(predicted) - static_cast<int64_t>(job.flipNs + step * (factor - 1u));
                if (off > -static_cast<int64_t>(period) && off < static_cast<int64_t>(2u * period + period / 2u))
                    showReal = predicted; // (else: a hitch or a pause - start again)
            }
            const uint64_t t0 = nowNs();
            m_xferLate.clear();
            m_hazard.reset();
            if (passes > 1u)
                findHazards(f);
            for (uint32_t pass = 1; pass <= passes; ++pass)
            {
                const float t = static_cast<float>(pass) / static_cast<float>(passes);
                const bool real = pass == passes;
                if (pass > 1u)
                    restoreHazardPages();
                static const bool s_verify = std::getenv("PS2_FRAME_INTERP_VERIFY") != nullptr;
                if (s_verify && m_vram)
                {
                    // debug: GS memory before each further pass against before the first one
                    m_inner->Sync(GSSyncReason::DebugReadback);
                    if (pass == 1u)
                        m_verify.assign(m_vram, m_vram + m_vramSize);
                    else
                    {
                        std::string list;
                        for (uint32_t page = 0; page < m_vramSize / kPageBytes; ++page)
                            if (std::memcmp(m_vram + page * kPageBytes, m_verify.data() + page * kPageBytes, kPageBytes) != 0)
                            {
                                if (!m_drawn.test(page))
                                    list += " " + std::to_string(page) + (m_hazard.test(page) ? "(restored!)" : "");
                            }
                        if (!list.empty())
                            std::fprintf(stderr, "[gs:interp] verify: pages still changed before pass %u:%s\n", pass, list.c_str());
                        // PS2_FRAME_INTERP_VERIFY=lo-hi (CPU renderer only): put those pages back too
                        unsigned lo = 0, hi = 0;
                        if (std::sscanf(std::getenv("PS2_FRAME_INTERP_VERIFY"), "%u-%u", &lo, &hi) == 2)
                        {
                            m_inner->Sync(GSSyncReason::DebugReadback);
                            for (uint32_t page = lo; page <= hi && page < m_vramSize / kPageBytes; ++page)
                                std::memcpy(m_vram + page * kPageBytes, m_verify.data() + page * kPageBytes, kPageBytes);
                        }
                    }
                }
                replay(f, prev, real ? f.doneOps : 0u, real ? 1.0f : t, pass == 1u && passes > 1u);
                if (pass == 1u && passes > 1u)
                {
                    m_statHazardPages += m_undoPages.size();
                    m_statHazardFrames += m_undoPages.empty() ? 0u : 1u;
                    m_statLate += m_statLateTransfers;
                    for (uint16_t page : m_undoPages)
                        m_statHazardDrawn += m_drawn.test(page) ? 1u : 0u;
                    if (s_log)
                    {
                        std::string list;
                        for (uint16_t page : m_undoPages)
                            list += " " + std::to_string(page) + (m_drawn.test(page) ? "*" : "");
                        std::fprintf(stderr, "[gs:interp] texture pages put back between passes (* in a buffer the frame draws into):%s; transfers left for the last pass: %u of %zu\n",
                                     list.c_str(), m_statLateTransfers, f.transfers.size());
                    }
                }
                if (pass == 1u)
                {
                    // The first picture is drawn (the GPU still needs a moment): against the
                    // time of a frame's first picture (also when this frame has only the one).
                    const uint64_t ready = nowNs() + m_marginNs;
                    const uint64_t first = showReal - step * (factor - 1u);
                    if (ready > first)
                    {
                        const uint64_t latest = job.flipNs + step * (factor - 1u) + 2u * period; // (beyond this, late it is)
                        const uint64_t moved = std::min(showReal + (ready - first), std::max(latest, showReal));
                        if (moved != showReal)
                        {
                            ++m_statPushed;
                            m_statPushedNs += moved - showReal;
                            m_statPushedMaxNs = std::max(m_statPushedMaxNs, moved - showReal);
                            if (moved - showReal > 2000000ull)
                                ++m_statPushedBig;
                            if (m_steady)
                                m_holdNs = std::min(m_holdNs + (moved - showReal), period / 2u);
                        }
                        showReal = moved;
                    }
                    else if (first - ready > m_slackNs + m_holdNs)
                        showReal -= std::min<uint64_t>((first - ready - m_slackNs - m_holdNs) / 32u, 100000ull);
                    m_holdNs -= m_holdNs / 512u;
                    m_statHoldNs += m_holdNs;
                    m_statDelayNs += showReal - job.flipNs;
                }
                s_nextDueNs.store(showReal - step * (passes - pass), std::memory_order_relaxed);
                m_inner->QueuePresentSnapshot(job.request);
                if (g_gsInterpPassHook)
                    g_gsInterpPassHook(m_inner.get(), pass, passes, real ? 1.0f : t);
            }
            m_lastShowReal = showReal;
            m_haveSchedule = true;
            const uint64_t t1 = nowNs();
            if (t1 - startNs > m_periodNs + m_periodNs / 3u)
            {
                char text[200];
                std::snprintf(text, sizeof(text), "drawing a frame's %u picture(s) took %.1f ms (%zu primitives; it had waited %.1f ms for the frame before)", passes,
                              (t1 - startNs) / 1e6, f.prims.size(), queued / 1e6);
                ps2HitchReport(text, startNs, t1);
            }

            ++m_statFlips;
            m_statPrims += f.prims.size();
            m_statMatched += f.matched;
            m_statRejected += f.rejected;
            m_statPictures += passes;
            m_statPassNs += t1 - t0;
            if (!usable && factor > 1u)
            {
                if (shed)
                    ++m_statShedFrames;
                else if (cut)
                    ++m_statCuts;
                else if (f.broken || !f.complete)
                    ++m_statBroken;
            }
            m_statRecovered += f.recovered;
            m_statCarried += f.carried;
            if (m_stats && m_statFlips >= 300u)
            {
                const double n = static_cast<double>(m_statFlips);
                std::fprintf(stderr, "[gs:interp] %ux: %.2f pictures per game frame; of %.0f primitives/frame %.1f%% matched, %.1f%% placed by the camera model or a neighbour, %.1f%% carried along, %.1f%% moved implausibly; matching %.2f ms/frame, drawing all passes %.2f ms/frame, game frame period %.2f ms, a frame reaches the render thread %.1f ms after the game's flip (%.2f ms of that waiting for the frame before); frames without in-between pictures: %llu cut, %llu needed the renderer mid-frame (%llu such calls), %llu while behind (%llu times); texture pages put back between passes: %.1f per frame (%.2f of them in drawn buffers), transfers left for the last pass: %.1f per frame; real picture shown %.1f ms after its flip on average, schedule moved back %llu times (%.1f ms in all, %llu of them by more than 2 ms, the largest %.1f ms), held back %.1f ms on average to keep it steady; game frames that took 1.5 periods or more: %llu (the longest %.1f ms)\n",
                             factor, static_cast<double>(m_statPictures) / n, static_cast<double>(m_statPrims) / n,
                             m_statPrims ? 100.0 * m_statMatched / m_statPrims : 0.0, m_statPrims ? 100.0 * m_statRecovered / m_statPrims : 0.0,
                             m_statPrims ? 100.0 * m_statCarried / m_statPrims : 0.0,
                             m_statPrims ? 100.0 * m_statRejected / m_statPrims : 0.0, m_statMatchNs / 1e6 / n, m_statPassNs / 1e6 / n, period / 1e6,
                             m_statLagNs / 1e6 / n, m_statQueuedNs / 1e6 / n, static_cast<unsigned long long>(m_statCuts), static_cast<unsigned long long>(m_statBroken),
                             static_cast<unsigned long long>(m_statDrains), static_cast<unsigned long long>(m_statShedFrames),
                             static_cast<unsigned long long>(m_statSheds), static_cast<double>(m_statHazardPages) / n, static_cast<double>(m_statHazardDrawn) / n,
                             static_cast<double>(m_statLate) / n, m_statDelayNs / 1e6 / n, static_cast<unsigned long long>(m_statPushed), m_statPushedNs / 1e6,
                             static_cast<unsigned long long>(m_statPushedBig), m_statPushedMaxNs / 1e6, m_statHoldNs / 1e6 / n,
                             static_cast<unsigned long long>(m_statLongFrames), m_statLongestNs / 1e6);
                m_statHazardPages = m_statHazardFrames = m_statLate = m_statHazardDrawn = m_statCarried = 0;
                m_statPushed = m_statDelayNs = m_statPushedNs = m_statQueuedNs = 0;
                m_statPushedBig = m_statPushedMaxNs = m_statHoldNs = m_statLongFrames = m_statLongestNs = 0;
                m_statFlips = m_statPrims = m_statMatched = m_statRejected = m_statPictures = m_statPassNs = m_statCuts = m_statBroken = m_statDrains = 0;
                m_statLagNs = m_statShedFrames = m_statSheds = 0;
                m_statMatchNs = m_statRecovered = 0;
            }

            noteLastClut(f);
        }

        // The palette load the next frame starts with: the last one of this frame that loads.
        void noteLastClut(const Frame &f)
        {
            for (size_t i = f.cluts.size(); i-- > 0u;)
            {
                const GSTex0Reg &t = f.cluts[i].tex0;
                if (!indexedFormat(t.psm) || t.cld == 0u || t.cld >= 6u)
                    continue;
                m_startClut = f.cluts[i];
                m_haveStartClut = true;
                return;
            }
        }

        // The game's frame period from its flips (smoothed; 1/60 s until known).
        uint64_t updatePeriod(uint64_t flipNs)
        {
            if (m_lastFlipNs != 0u)
            {
                const uint64_t d = flipNs - m_lastFlipNs;
                // (for the log: game frames that took a period and a half or more - a frame the
                // game lost, not one that was merely handed over late within its period)
                if (d > m_periodNs + m_periodNs / 2u && d < 100000000ull)
                {
                    ++m_statLongFrames;
                    m_statLongestNs = std::max(m_statLongestNs, d);
                }
                if (d > 4000000ull && d < 100000000ull)
                    m_periodNs = (m_periodNs * 15u + d) / 16u;
            }
            m_lastFlipNs = flipNs;
            return m_periodNs;
        }

        std::unique_ptr<GSRasterBackend> m_inner;
        GSRasterBackendEx *m_ex = nullptr; // m_inner's extra calls, if it has them
        SpinLock m_mutex;
        // Frames not in use (their vectors keep their capacity). Declared before the frames:
        // those go back in here when released.
        std::mutex m_poolMutex;
        std::vector<std::unique_ptr<Frame>> m_pool;
        std::shared_ptr<Frame> m_cur, m_last; // being recorded; the one before (GS thread)
        std::vector<uint8_t> m_prevTaken;     // per object of m_last: paired with one of m_cur
        uint64_t m_matchCount = 0;
        // the render thread and the frame waiting for it
        bool m_threaded = true;
        std::thread m_thread;
        std::mutex m_jobMutex;
        std::condition_variable m_jobCv;
        Job m_job;
        bool m_jobQueued = false, m_jobRunning = false, m_quit = false;
        bool m_active = false;
        bool m_runOpen = false;
        bool m_transferBegun = false; // a transfer was started since the last flip
        // read-backs without waiting (PS2_ASYNC_READS=0: the renderer waits for each as before)
        bool m_asyncReads = !(std::getenv("PS2_ASYNC_READS") && std::getenv("PS2_ASYNC_READS")[0] == '0');
        std::atomic<bool> m_haveHeld{false};
        GSTransferCommand m_held{};
        uint64_t m_kickBase = 0;
        struct Pair
        {
            uint32_t cur, prev;
            bool sameData;
        };
        struct Candidate
        {
            float distance;
            uint32_t cur, prev;
        };
        std::vector<Pair> m_pairs;
        struct Implausible
        {
            uint32_t object, prim, partner; // object and primitive of this frame, the partner's first vertex
            bool wide;                      // within the wider limits for things that move by themselves
        };
        std::vector<Implausible> m_implausible;
        std::vector<uint32_t> m_movers, m_moverParent, m_groupPrims, m_groupHeld;
        struct CornerEntry
        {
            uint64_t key = 0;
            uint32_t mover = 0;
            bool used = false;
        };
        std::vector<CornerEntry> m_cornerTable;
        std::vector<CornerEntry> m_layerTable;  // releaseWorldLayers
        std::vector<uint32_t> m_layerObjects;
        std::vector<Candidate> m_candidates;
        std::vector<uint8_t> m_curTaken;
        std::vector<uint32_t> m_restCur, m_restPrev;
        struct TableEntry
        {
            uint64_t key = 0;
            uint32_t prev = kNone;
            float ratio = 1.0f;
            float zCur = 0.0f;
            float qCur = 0.0f;
            uint32_t object = 0; // whose corner it is
            bool used = false;
            bool persp = false;
        };
        struct Work
        {
            uint32_t prim, object;
            uint8_t farMask;
            bool inScene, flat;
        };
        std::vector<Work> m_work;
        std::vector<uint64_t> m_filter;
        std::vector<TableEntry> m_table;
        struct GridCell
        {
            float dx = 0.0f, dy = 0.0f, n = 0.0f;
        };
        std::vector<GridCell> m_grid;
        std::vector<uint32_t> m_gridQueue;
        std::vector<uint32_t> m_progPc;       // VU1 programs seen (Object::prog indexes this)
        std::vector<ProgFrame> m_progFrame;   // per program, this frame
        std::vector<uint16_t> m_progClips;    // per program: frames left for which it is known to clip (fitCamera)
        std::vector<uint32_t> m_progPairs;    // per program, this frame: paired objects, and those whose count changed
        std::vector<Sample> m_samples;
        std::vector<uint8_t> m_uvIndex; // by texture address: 1 + index in the frame's uvModels (fitTextureModels)
        std::vector<uint64_t> m_uvSkipUntil; // by texture address: the match count before which it is not looked at
        std::vector<Normal> m_uvNormals;
        std::vector<uint32_t> m_uvCounts, m_uvFits, m_uvIndexed;
        std::vector<float> m_uvChange;
        double m_model[3][4]{};
        double m_skyModel[3][4]{}; // ... for what turns with the camera at no distance (the sky)
        float m_skySlope = 0.0f;   // the sky's depth per unit of q (kept from frame to frame)
        bool m_haveModel = false;
        float m_modelShare = 0.0f; // of the sampled matched vertices, how many the camera model explains
        float m_zSlope = 0.0f;
        uint32_t m_testCount = 0;
        bool m_probe = false;
        float m_probeX = 0.0f, m_probeY = 0.0f;
        uint8_t *m_vram = nullptr;
        uint32_t m_vramSize = 0;
        PageSet m_hazard;
        std::vector<uint8_t> m_verify;
        ClutLoad m_startClut{};
        bool m_haveStartClut = false;
        bool m_needStartClut = false; // this frame draws with it before loading one of its own
        bool m_noClutRestore = std::getenv("PS2_FRAME_INTERP_NOCLUT") != nullptr;
        std::vector<uint8_t> m_undo;
        std::vector<uint16_t> m_undoPages;
        PageSet m_drawn;                 // pages this frame draws into
        PageSet m_saved;                 // hazard pages kept so far (first pass)
        std::vector<uint8_t> m_xferLate; // per transfer of the frame: nothing reads it before the frame ends
        uint32_t m_statLateTransfers = 0;
        uint64_t m_statLate = 0, m_statHazardDrawn = 0, m_statCarried = 0;
        bool m_hazardLog = std::getenv("PS2_FRAME_INTERP_HAZARDLOG") != nullptr;
        uint64_t m_statHazardPages = 0, m_statHazardFrames = 0;
        uint64_t m_flipCount = 0, m_shedUntil = 0, m_shedFrames = 30, m_statLagNs = 0, m_statQueuedNs = 0, m_statShedFrames = 0, m_statSheds = 0;
        bool m_shedding = !(std::getenv("PS2_FRAME_INTERP_SHED") && std::getenv("PS2_FRAME_INTERP_SHED")[0] == '0');
        float m_maxMove = 96.0f;
        uint32_t m_dbgSkip = std::getenv("PS2_FRAME_INTERP_SKIP") ? static_cast<uint32_t>(std::atoi(std::getenv("PS2_FRAME_INTERP_SKIP"))) : 0u;
        uint32_t m_dbgOnly = std::getenv("PS2_FRAME_INTERP_ONLY") ? static_cast<uint32_t>(std::atoi(std::getenv("PS2_FRAME_INTERP_ONLY"))) : 0u;

        uint64_t m_lastFlipNs = 0, m_periodNs = 16683333ull, m_lastShowReal = 0;
        bool m_haveSchedule = false;
        // a drawn picture is taken to be ready this much later (the GPU's share); slack kept before its time
        uint64_t m_marginNs = static_cast<uint64_t>((std::getenv("PS2_FRAME_INTERP_MARGIN_MS") ? std::max(0.0, std::atof(std::getenv("PS2_FRAME_INTERP_MARGIN_MS"))) : 3.0) * 1e6);
        uint64_t m_slackNs = 1000000ull;
        uint64_t m_statPushed = 0, m_statPushedNs = 0, m_statDelayNs = 0;
        uint64_t m_statPushedBig = 0, m_statPushedMaxNs = 0, m_statHoldNs = 0, m_statLongFrames = 0, m_statLongestNs = 0;
        uint64_t m_holdNs = 0; // what pushes added to the schedule lately and is not given back yet (see runJob)
        bool m_steady = std::getenv("PS2_FRAME_INTERP_STEADY") && std::getenv("PS2_FRAME_INTERP_STEADY")[0] == '1';

        bool m_stats = false;
        uint64_t m_statFlips = 0, m_statPrims = 0, m_statMatched = 0, m_statRejected = 0, m_statPictures = 0, m_statPassNs = 0, m_statMatchNs = 0, m_statRecovered = 0,
                 m_statCuts = 0, m_statBroken = 0, m_statDrains = 0;
    };
}

std::unique_ptr<GSRasterBackend> ps2CreateInterpGsBackend(std::unique_ptr<GSRasterBackend> inner, GSRasterBackendEx *ex)
{
    if (!inner)
        return inner;
    return std::make_unique<GsInterpBackend>(std::move(inner), ex);
}

void ps2GsInterpSetFactor(uint32_t factor)
{
    s_wantFactor.store(std::clamp<uint32_t>(factor, 1u, kMaxFactor));
}

uint32_t ps2GsInterpFactor()
{
    // 1 until the layer exists (no tags, no deferred reads without it)
    const uint32_t f = s_wantFactor.load(std::memory_order_relaxed);
    return f == 0u ? 1u : f;
}

void ps2GsInterpObjectTag(uint32_t pc, uint32_t hash)
{
    if (s_instance)
        s_instance->objectTag(pc, hash);
}

bool ps2GsInterpDeferReadback(uint32_t bytes, std::function<void(std::vector<uint8_t> &&data, uint32_t got)> done)
{
    return s_instance && s_instance->deferReadback(bytes, std::move(done));
}

uint64_t ps2GsInterpNextPictureDueNs()
{
    return s_nextDueNs.load(std::memory_order_relaxed);
}
