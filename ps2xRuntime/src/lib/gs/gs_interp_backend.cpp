#include "runtime/gs/gs_interp_backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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
    };

    struct Prim
    {
        uint32_t vtx = 0;     // first vertex in Frame::verts
        uint32_t prev = kNone; // first vertex of the matching primitive in the previous frame
        uint32_t kick = 0;    // position in its object's output
        uint8_t count = 0;
    };

    struct Object
    {
        uint64_t key = 0; // pc | hash << 32
        uint32_t primStart = 0, primCount = 0;
        float sumX = 0.0f, sumY = 0.0f; // of its vertices (for "which instance is which")
        uint32_t vertices = 0;
        bool taken = false; // matched by an object of the next frame

        float cx() const { return vertices ? sumX / static_cast<float>(vertices) : 0.0f; }
        float cy() const { return vertices ? sumY / static_cast<float>(vertices) : 0.0f; }
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
    };

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
        size_t doneOps = 0;     // ops already executed (a call needed the renderer's state)
        bool broken = false;    // ... and they included drawing or transfers: no in-between pictures
        bool complete = false;  // recorded from flip to flip without a gap
        uint32_t matched = 0, rejected = 0, recovered = 0;

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
            doneOps = 0;
            broken = false;
            complete = false;
            matched = rejected = recovered = 0;
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
        explicit GsInterpBackend(std::unique_ptr<GSRasterBackend> inner)
            : m_inner(std::move(inner)), m_maxMove(envMaxMove())
        {
            if (s_wantFactor.load() == 0u)
                s_wantFactor.store(envFactor());
            m_stats = std::getenv("PS2_GS_VK_STATS") != nullptr || std::getenv("PS2_FRAME_INTERP_STATS") != nullptr;
            m_active = s_wantFactor.load() > 1u;
            m_cur.complete = true;
            s_instance = this;
            std::fprintf(stderr, "[gs:interp] frame interpolation layer: %u picture(s) per game frame (PS2_FRAME_INTERP=1..%u, F7 in game)\n",
                         s_wantFactor.load(), kMaxFactor);
        }
        ~GsInterpBackend() override
        {
            if (s_instance == this)
                s_instance = nullptr;
        }

        void Initialize(uint8_t *vram, uint32_t vramSize) override
        {
            std::lock_guard<SpinLock> lock(m_mutex);
            drain();
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
                m_inner->LoadClut(tex0, texclut);
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            m_cur.cluts.push_back({tex0, texclut});
            pushOp(OpKind::LoadClut, static_cast<uint32_t>(m_cur.cluts.size() - 1u));
        }
        void BeginTransfer(const GSTransferCommand &command) override
        {
            if (!m_active)
            {
                m_inner->BeginTransfer(command);
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            m_cur.transfers.push_back(command);
            pushOp(OpKind::Transfer, static_cast<uint32_t>(m_cur.transfers.size() - 1u));
            m_transferBegun = true;
        }
        void UploadImage(const uint8_t *data, uint32_t sizeBytes) override
        {
            if (!m_active)
            {
                m_inner->UploadImage(data, sizeBytes);
                return;
            }
            std::lock_guard<SpinLock> lock(m_mutex);
            // Image data of a transfer that began before the last flip: it can only be sent to
            // the renderer once, so this frame is drawn once.
            if (!m_transferBegun)
                m_cur.broken = true;
            const size_t at = m_cur.bytes.size();
            m_cur.bytes.insert(m_cur.bytes.end(), data, data + sizeBytes);
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
            if (m_cur.ops.empty() || m_cur.ops.back().kind != OpKind::Flush)
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
                return false;
            std::lock_guard<SpinLock> lock(m_mutex);
            m_cur.readbacks.push_back({bytes, std::move(done)});
            pushOp(OpKind::Readback, static_cast<uint32_t>(m_cur.readbacks.size() - 1u));
            return true;
        }

    private:
        void pushOp(OpKind kind, uint32_t a = 0, uint32_t b = 0)
        {
            m_cur.ops.push_back({kind, a, b});
            m_runOpen = false;
        }

        void beginObject(uint64_t key)
        {
            Frame &f = m_cur;
            closeObject();
            Object obj;
            obj.key = key;
            obj.primStart = static_cast<uint32_t>(f.prims.size());
            f.objects.push_back(obj);
            m_kickBase = g_ssx3GsKicks.load(std::memory_order_relaxed);
        }

        void closeObject()
        {
            Frame &f = m_cur;
            if (f.objects.empty())
                return;
            Object &obj = f.objects.back();
            obj.primCount = static_cast<uint32_t>(f.prims.size()) - obj.primStart;
            if (obj.primCount == 0u)
                f.objects.pop_back();
        }

        void recordPrim(const GSPrimitiveBatch &batch)
        {
            Frame &f = m_cur;
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
            for (uint32_t i = 0; i < prim.count; ++i)
            {
                f.pos.push_back({batch.vertices[i].x, batch.vertices[i].y});
                obj.sumX += batch.vertices[i].x;
                obj.sumY += batch.vertices[i].y;
            }
            obj.vertices += prim.count;
            f.prims.push_back(prim);
            ++f.runs.back().primCount;
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
        // and its shape (the edges from the first vertex) about the same.
        inline bool plausibleMove(const XY *c, const XY *p, uint32_t count) const
        {
            for (uint32_t v = 0; v < count; ++v)
                if (std::fabs(c[v].x - p[v].x) > m_maxMove || std::fabs(c[v].y - p[v].y) > m_maxMove)
                    return false;
            for (uint32_t v = 1; v < count; ++v)
            {
                const float cex = c[v].x - c[0].x, cey = c[v].y - c[0].y, pex = p[v].x - p[0].x, pey = p[v].y - p[0].y;
                const float longest = std::max(std::max(std::fabs(cex), std::fabs(cey)), std::max(std::fabs(pex), std::fabs(pey)));
                const float allowed = 0.5f * longest + 4.0f;
                if (std::fabs(cex - pex) > allowed || std::fabs(cey - pey) > allowed)
                    return false;
            }
            return true;
        }

        // Pair the primitives of object `co` (this frame) with those of `po` (last frame). Both
        // come out of the same program in the same order, except where triangles were clipped
        // (a clipped one becomes a different number of primitives): walk both lists and, where
        // they stop agreeing, look a few primitives ahead in either for the next agreement.
        void matchPrims(const Object &co, const Object &po)
        {
            Frame &f = m_cur;
            Prim *cp = f.prims.data() + co.primStart;
            const Prim *pp = m_prev.prims.data() + po.primStart;
            const GSVertex *cverts = f.verts.data(), *pverts = m_prev.verts.data();
            const XY *cpos = f.pos.data(), *ppos = m_prev.pos.data();
            constexpr uint32_t kWindow = 8u;
            // The usual case: nothing clipped, both lists have the same primitives at the same
            // places in the program's output - pair them in order.
            if (co.primCount == po.primCount)
            {
                bool identical = true;
                for (uint32_t k = 0; k < co.primCount && identical; ++k)
                    identical = cp[k].kick == pp[k].kick && cp[k].count == pp[k].count;
                if (identical)
                {
                    for (uint32_t k = 0; k < co.primCount; ++k)
                    {
                        if (plausibleMove(cpos + cp[k].vtx, ppos + pp[k].vtx, cp[k].count))
                        {
                            cp[k].prev = pp[k].vtx;
                            ++f.matched;
                        }
                        else
                            ++f.rejected; // may still be placed from its neighbours
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
                if (plausibleMove(cpos + cp[i].vtx, ppos + pp[j].vtx, cp[i].count))
                {
                    cp[i].prev = pp[j].vtx;
                    ++f.matched;
                }
                else
                    ++f.rejected; // may still be placed from its neighbours
                ++i;
                ++j;
            }
        }

        // An object that turns or deforms a lot in one frame (the trick meter's spinning coil,
        // some effects) cannot be moved in straight lines: triangles pass through each other and
        // collapse on the way. Signs of that: triangles that face the other way than last frame,
        // or that all but vanish half-way. If more than a tenth of the object's triangles show
        // them, the object keeps this frame's positions in the in-between pictures.
        void checkObject(const Object &co)
        {
            Frame &f = m_cur;
            Prim *cp = f.prims.data() + co.primStart;
            uint32_t triangles = 0, bad = 0;
            for (uint32_t i = 0; i < co.primCount; ++i)
            {
                if (!hasPrev(cp[i].prev) || cp[i].count != 3u)
                    continue;
                const XY *c = f.pos.data() + cp[i].vtx;
                const XY *q = m_prev.pos.data() + cp[i].prev;
                const float ac = (c[1].x - c[0].x) * (c[2].y - c[0].y) - (c[2].x - c[0].x) * (c[1].y - c[0].y);
                const float ap = (q[1].x - q[0].x) * (q[2].y - q[0].y) - (q[2].x - q[0].x) * (q[1].y - q[0].y);
                const float smaller = std::min(std::fabs(ac), std::fabs(ap));
                if (smaller < 4.0f)
                    continue; // slivers flip and vanish all the time
                ++triangles;
                const float mx0 = (c[0].x + q[0].x) * 0.5f, my0 = (c[0].y + q[0].y) * 0.5f;
                const float am = ((c[1].x + q[1].x) * 0.5f - mx0) * ((c[2].y + q[2].y) * 0.5f - my0) - ((c[2].x + q[2].x) * 0.5f - mx0) * ((c[1].y + q[1].y) * 0.5f - my0);
                if ((ac > 0.0f) != (ap > 0.0f) || std::fabs(am) < 0.25f * smaller)
                    ++bad;
            }
            static const bool s_dbg = std::getenv("PS2_FRAME_INTERP_DEBUG") != nullptr;
            if (s_dbg && (bad != 0u || std::getenv("PS2_FRAME_INTERP_DEBUG")[0] == '2'))
                std::fprintf(stderr, "[gs:interp] object %llx at %.0f,%.0f: %u prims, %u sizeable matched triangles, %u flipped or collapsing\n",
                             static_cast<unsigned long long>(co.key), co.cx(), co.cy(), co.primCount, triangles, bad);
            if (bad < 3u || bad * 10u <= triangles)
                return;
            for (uint32_t i = 0; i < co.primCount; ++i)
            {
                if (hasPrev(cp[i].prev))
                {
                    --f.matched;
                    ++f.rejected;
                }
                cp[i].prev = kRejected; // the whole object, also what found no partner
            }
        }

        // Two things would open cracks between neighbouring triangles in the in-between pictures:
        //  - primitives without a partner (mostly triangles the VU1 program clipped into a
        //    different number of pieces than last frame; also models that just appeared) would
        //    stay at this frame's position while their neighbours move;
        //  - the far-away corners that clipping makes sit on the edge of the GS coordinate range
        //    and are different points of the triangle every frame, so two triangles that share
        //    one disagree on where it was.
        // So every corner gets its old position by one rule that depends only on the corner:
        //  - on or near the screen, and a corner of a matched primitive anywhere in the frame:
        //    that corner's old position (for a matched primitive, its own);
        //  - otherwise: moved by what the picture did around it - the average motion of the
        //    matched primitives in the same 64-pixel square of GS coordinates, or the nearest
        //    square that has any.
        // A primitive without a partner and without any shared corner stays where it is.
        void recoverUnmatched()
        {
            Frame &f = m_cur;
            if (f.matched == 0u)
                return;
            uint32_t unmatched = 0;
            for (const Prim &p : f.prims)
                unmatched += p.prev == kNone ? 1u : 0u;
            uint32_t capacity = 64u;
            while (capacity < unmatched * 6u)
                capacity <<= 1;
            m_table.assign(capacity, TableEntry{});
            const uint32_t mask = capacity - 1u;
            auto keyOf = [](const XY &v)
            {
                uint32_t xb, yb;
                std::memcpy(&xb, &v.x, 4);
                std::memcpy(&yb, &v.y, 4);
                return (static_cast<uint64_t>(xb) << 32) | yb;
            };
            auto slotOf = [mask](uint64_t key)
            {
                return static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ull) >> 40) & mask;
            };
            // the corners we are looking for
            for (const Prim &p : f.prims)
            {
                if (p.prev != kNone)
                    continue;
                for (uint32_t v = 0; v < p.count; ++v)
                {
                    const uint64_t key = keyOf(f.pos[p.vtx + v]);
                    uint32_t slot = slotOf(key);
                    while (m_table[slot].used && m_table[slot].key != key)
                        slot = (slot + 1u) & mask;
                    if (!m_table[slot].used)
                        m_table[slot] = TableEntry{key, kNone, true};
                }
            }
            // One pass over the runs: the matched primitives give the wanted corners their old
            // positions and fill the motion squares (near corners only); primitives with far
            // corners are noted.
            m_grid.assign(kGridCells * kGridCells, GridCell{});
            m_farPrims.clear();
            for (const Run &run : f.runs)
            {
                const GSContext &ctx = f.states[run.state].context;
                const float margin = 96.0f;
                const float wx0 = static_cast<float>(ctx.xyoffset.ofx >> 4) + static_cast<float>(ctx.scissor.x0) - margin;
                const float wx1 = static_cast<float>(ctx.xyoffset.ofx >> 4) + static_cast<float>(ctx.scissor.x1) + margin;
                const float wy0 = static_cast<float>(ctx.xyoffset.ofy >> 4) + static_cast<float>(ctx.scissor.y0) - margin;
                const float wy1 = static_cast<float>(ctx.xyoffset.ofy >> 4) + static_cast<float>(ctx.scissor.y1) + margin;
                Prim *p = f.prims.data() + run.primStart;
                for (uint32_t k = 0; k < run.primCount; ++k, ++p)
                {
                    if (p->prev == kRejected)
                        continue;
                    const XY *c = f.pos.data() + p->vtx;
                    uint8_t farMask = 0u;
                    for (uint32_t v = 0; v < p->count; ++v)
                        if (c[v].x < wx0 || c[v].x > wx1 || c[v].y < wy0 || c[v].y > wy1)
                            farMask |= static_cast<uint8_t>(1u << v);
                    if (farMask != 0u)
                        m_farPrims.push_back({static_cast<uint32_t>(p - f.prims.data()), farMask});
                    if (p->prev == kNone)
                        continue;
                    const XY *q = m_prev.pos.data() + p->prev;
                    for (uint32_t v = 0; v < p->count; ++v)
                    {
                        if (farMask & (1u << v))
                            continue;
                        if (unmatched != 0u)
                        {
                            const uint64_t key = keyOf(c[v]);
                            uint32_t slot = slotOf(key);
                            while (m_table[slot].used && m_table[slot].key != key)
                                slot = (slot + 1u) & mask;
                            if (m_table[slot].used && m_table[slot].prev == kNone)
                                m_table[slot].prev = p->prev + v;
                        }
                        if (v == 0u)
                        {
                            GridCell &cell = m_grid[gridIndex(c[0].x, c[0].y)];
                            cell.dx += q[0].x - c[0].x;
                            cell.dy += q[0].y - c[0].y;
                            cell.n += 1.0f;
                        }
                    }
                }
            }
            if (unmatched == 0u && m_farPrims.empty())
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

            // matched primitives with far corners: those corners by the squares
            for (const FarPrim &fp : m_farPrims)
            {
                Prim &p = f.prims[fp.prim];
                if (!hasPrev(p.prev))
                    continue;
                const GSVertex *cv = f.verts.data() + p.vtx;
                GSVertex old[3];
                double dz = 0.0;
                uint32_t nearCount = 0;
                for (uint32_t v = 0; v < p.count; ++v)
                {
                    old[v] = m_prev.verts[p.prev + v];
                    if (!(fp.farMask & (1u << v)))
                    {
                        dz += old[v].z - cv[v].z;
                        ++nearCount;
                    }
                }
                if (nearCount != 0u)
                    dz /= static_cast<double>(nearCount);
                for (uint32_t v = 0; v < p.count; ++v)
                {
                    if (!(fp.farMask & (1u << v)))
                        continue;
                    const GridCell &cell = m_grid[gridIndex(cv[v].x, cv[v].y)];
                    old[v].x = cv[v].x + cell.dx;
                    old[v].y = cv[v].y + cell.dy;
                    if (nearCount != 0u)
                        old[v].z = std::max(0.0, cv[v].z + dz);
                }
                p.prev = kSynth | static_cast<uint32_t>(m_synth.size());
                m_synth.insert(m_synth.end(), old, old + p.count);
            }

            // primitives without a partner
            size_t farCursor = 0;
            for (uint32_t index = 0; index < f.prims.size() && unmatched != 0u; ++index)
            {
                Prim &p = f.prims[index];
                while (farCursor < m_farPrims.size() && m_farPrims[farCursor].prim < index)
                    ++farCursor;
                if (p.prev != kNone)
                    continue;
                const uint8_t farMask = farCursor < m_farPrims.size() && m_farPrims[farCursor].prim == index ? m_farPrims[farCursor].farMask : uint8_t{0};
                const GSVertex *cv = f.verts.data() + p.vtx;
                GSVertex old[3];
                bool found[3] = {false, false, false};
                uint32_t foundCount = 0;
                double dz = 0.0;
                for (uint32_t v = 0; v < p.count; ++v)
                {
                    old[v] = cv[v];
                    if (farMask & (1u << v))
                        continue;
                    const uint64_t key = keyOf(f.pos[p.vtx + v]);
                    uint32_t slot = slotOf(key);
                    while (m_table[slot].used && m_table[slot].key != key)
                        slot = (slot + 1u) & mask;
                    if (!m_table[slot].used || m_table[slot].prev == kNone)
                        continue;
                    const GSVertex &pv = m_prev.verts[m_table[slot].prev];
                    old[v].x = pv.x;
                    old[v].y = pv.y;
                    old[v].z = pv.z;
                    dz += pv.z - cv[v].z;
                    found[v] = true;
                    ++foundCount;
                }
                if (foundCount == 0u)
                    continue;
                dz /= static_cast<double>(foundCount);
                for (uint32_t v = 0; v < p.count; ++v)
                {
                    if (found[v])
                        continue;
                    const GridCell &cell = m_grid[gridIndex(cv[v].x, cv[v].y)];
                    old[v].x += cell.dx;
                    old[v].y += cell.dy;
                    old[v].z = std::max(0.0, old[v].z + dz);
                }
                p.prev = kSynth | static_cast<uint32_t>(m_synth.size());
                m_synth.insert(m_synth.end(), old, old + p.count);
                ++f.recovered;
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
        void matchGroup(const uint32_t *cur, size_t curCount, const uint32_t *prev, size_t prevCount, bool sameData)
        {
            Frame &f = m_cur;
            if (curCount == 1u && prevCount == 1u && sameData)
            {
                m_prev.objects[prev[0]].taken = true;
                m_pairs.push_back({cur[0], prev[0]});
                return;
            }
            m_candidates.clear();
            const float limit = m_maxMove * m_maxMove;
            for (size_t i = 0; i < curCount; ++i)
            {
                const Object &co = f.objects[cur[i]];
                for (size_t j = 0; j < prevCount; ++j)
                {
                    const Object &po = m_prev.objects[prev[j]];
                    if (po.taken || (!sameData && po.primCount != co.primCount))
                        continue;
                    const float dx = co.cx() - po.cx(), dy = co.cy() - po.cy();
                    const float d = dx * dx + dy * dy;
                    if (d <= limit)
                        m_candidates.push_back({d, cur[i], prev[j]});
                }
            }
            std::sort(m_candidates.begin(), m_candidates.end(), [](const Candidate &a, const Candidate &b)
                      { return a.distance < b.distance; });
            for (const Candidate &c : m_candidates)
            {
                if (m_curTaken[c.cur] || m_prev.objects[c.prev].taken)
                    continue;
                m_curTaken[c.cur] = 1u;
                m_prev.objects[c.prev].taken = true;
                m_pairs.push_back({c.cur, c.prev});
            }
        }

        // At the flip: which object of last frame is each object of this frame?
        //  1. same program and same input data (a model drawn again): by nearness when it is
        //     drawn several times (instances);
        //  2. what is left, same program and same number of primitives (animated models,
        //     particles: the data changes every frame): by nearness.
        void matchFrame()
        {
            const uint64_t tStart = nowNs();
            m_phaseNs[0] = m_phaseNs[1] = m_phaseNs[2] = 0;
            Frame &f = m_cur;
            f.byKey.resize(f.objects.size());
            for (uint32_t i = 0; i < f.byKey.size(); ++i)
                f.byKey[i] = i;
            std::sort(f.byKey.begin(), f.byKey.end(), [&](uint32_t a, uint32_t b)
                      { return f.objects[a].key != f.objects[b].key ? f.objects[a].key < f.objects[b].key : a < b; });
            m_synth.clear();
            if (m_prev.objects.empty())
                return;
            m_pairs.clear();
            m_curTaken.assign(f.objects.size(), 0u);
            for (Object &o : m_prev.objects)
                o.taken = false;

            // 1. equal keys
            size_t i = 0, j = 0;
            const std::vector<uint32_t> &ck = f.byKey, &pk = m_prev.byKey;
            while (i < ck.size() && j < pk.size())
            {
                const uint64_t a = f.objects[ck[i]].key, b = m_prev.objects[pk[j]].key;
                if (a < b)
                    ++i;
                else if (b < a)
                    ++j;
                else
                {
                    size_t ie = i, je = j;
                    while (ie < ck.size() && f.objects[ck[ie]].key == a)
                        ++ie;
                    while (je < pk.size() && m_prev.objects[pk[je]].key == a)
                        ++je;
                    const size_t before = m_pairs.size();
                    matchGroup(ck.data() + i, ie - i, pk.data() + j, je - j, true);
                    for (size_t k = before; k < m_pairs.size(); ++k)
                        m_curTaken[m_pairs[k].cur] = 1u;
                    i = ie;
                    j = je;
                }
            }

            // 2. the rest, by program (the low half of the key)
            m_restCur.clear();
            m_restPrev.clear();
            for (uint32_t k = 0; k < f.objects.size(); ++k)
                if (!m_curTaken[k])
                    m_restCur.push_back(k);
            for (uint32_t k = 0; k < m_prev.objects.size(); ++k)
                if (!m_prev.objects[k].taken)
                    m_restPrev.push_back(k);
            auto byProgram = [](const std::vector<Object> &objects)
            {
                return [&objects](uint32_t a, uint32_t b)
                {
                    const uint32_t pa = static_cast<uint32_t>(objects[a].key), pb = static_cast<uint32_t>(objects[b].key);
                    return pa != pb ? pa < pb : a < b;
                };
            };
            std::sort(m_restCur.begin(), m_restCur.end(), byProgram(f.objects));
            std::sort(m_restPrev.begin(), m_restPrev.end(), byProgram(m_prev.objects));
            i = j = 0;
            while (i < m_restCur.size() && j < m_restPrev.size())
            {
                const uint32_t a = static_cast<uint32_t>(f.objects[m_restCur[i]].key), b = static_cast<uint32_t>(m_prev.objects[m_restPrev[j]].key);
                if (a < b)
                    ++i;
                else if (b < a)
                    ++j;
                else
                {
                    size_t ie = i, je = j;
                    while (ie < m_restCur.size() && static_cast<uint32_t>(f.objects[m_restCur[ie]].key) == a)
                        ++ie;
                    while (je < m_restPrev.size() && static_cast<uint32_t>(m_prev.objects[m_restPrev[je]].key) == a)
                        ++je;
                    matchGroup(m_restCur.data() + i, ie - i, m_restPrev.data() + j, je - j, false);
                    i = ie;
                    j = je;
                }
            }

            const uint64_t t0 = nowNs();
            for (const Pair &pair : m_pairs)
            {
                matchPrims(f.objects[pair.cur], m_prev.objects[pair.prev]);
                checkObject(f.objects[pair.cur]);
            }
            const uint64_t t1 = nowNs();
            recoverUnmatched();
            m_phaseNs[0] = t0 - tStart;
            m_phaseNs[1] = t1 - t0;
            m_phaseNs[2] = nowNs() - t1;
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
        void replay(size_t from, float t)
        {
            Frame &f = m_cur;
            const bool between = t < 1.0f;
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
                    const Prim *p = f.prims.data() + run.primStart;
                    for (uint32_t k = 0; k < run.primCount; ++k, ++p)
                    {
                        const GSVertex *cv = f.verts.data() + p->vtx;
                        batch.vertexCount = p->count;
                        if (between && hasPrev(p->prev) && (m_dbgOnly == 0u || p->count == m_dbgOnly))
                        {
                            const GSVertex *pv = (p->prev & kSynth) ? m_synth.data() + (p->prev & ~kSynth) : m_prev.verts.data() + p->prev;
                            for (uint32_t v = 0; v < p->count; ++v)
                            {
                                GSVertex &o = batch.vertices[v];
                                const GSVertex &a = pv[v];
                                const GSVertex &b = cv[v];
                                o.x = a.x + (b.x - a.x) * t;
                                o.y = a.y + (b.y - a.y) * t;
                                o.z = a.z + (b.z - a.z) * static_cast<double>(t);
                                // Only the position moves. Colour, fog and texture coordinates stay
                                // this frame's: they may scroll, wrap or pulse from frame to frame,
                                // and a wrong pairing of two things at the same place then does no
                                // harm. The perspective term follows the vertex.
                                o.q = a.q + (b.q - a.q) * t;
                                const float ratio = b.q != 0.0f ? o.q / b.q : 1.0f;
                                o.s = b.s * ratio;
                                o.t = b.t * ratio;
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
                        m_inner->Submit(batch);
                    }
                    break;
                }
                case OpKind::LoadClut:
                    m_inner->LoadClut(f.cluts[op.a].tex0, f.cluts[op.a].texclut);
                    break;
                case OpKind::Transfer:
                    // A local->host transfer only feeds a read-back: real frame only.
                    skipUpload = between && f.transfers[op.a].direction == 1u;
                    if (!skipUpload)
                        m_inner->BeginTransfer(f.transfers[op.a]);
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
                    if (!between)
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
            Frame &f = m_cur;
            if (f.doneOps >= f.ops.size())
                return;
            bool drawing = false;
            for (size_t i = f.doneOps; i < f.ops.size(); ++i)
                drawing = drawing || (f.ops[i].kind != OpKind::Flush && f.ops[i].kind != OpKind::TextureFlush);
            replay(f.doneOps, 1.0f);
            f.doneOps = f.ops.size();
            m_runOpen = false;
            if (drawing)
            {
                f.broken = true;
                ++m_statDrains;
            }
        }

        void flip(const GSPresentationRequest &request)
        {
            const uint64_t flipNs = nowNs();
            if (!m_active)
            {
                s_nextDueNs.store(0u, std::memory_order_relaxed);
                m_inner->QueuePresentSnapshot(request);
                if (g_gsInterpPassHook)
                    g_gsInterpPassHook(m_inner.get(), 1u, 1u, 1.0f);
                m_active = s_wantFactor.load(std::memory_order_relaxed) > 1u;
                m_cur.clear();
                m_cur.complete = m_active;
                m_prev.clear();
                m_haveSchedule = false;
                return;
            }
            Frame &f = m_cur;
            closeObject();
            if (f.prims.empty())
            {
                // A flip with nothing drawn since the last one (a second display register, a
                // paused game): show it, and keep the last drawn frame as "last frame".
                replay(f.doneOps, 1.0f);
                s_nextDueNs.store(m_haveSchedule ? m_lastShowReal + 1u : 0u, std::memory_order_relaxed);
                m_inner->QueuePresentSnapshot(request);
                if (g_gsInterpPassHook)
                    g_gsInterpPassHook(m_inner.get(), 1u, 1u, 1.0f);
                g_ps2FlipIssuedNs.store(0u, std::memory_order_relaxed);
                m_cur.clear();
                m_runOpen = false;
                m_transferBegun = false;
                m_active = s_wantFactor.load(std::memory_order_relaxed) > 1u;
                m_cur.complete = m_active;
                if (!m_active)
                    m_prev.clear();
                return;
            }
            matchFrame();
            const uint64_t matchNs = nowNs() - flipNs;
            m_statMatchNs += matchNs;
            const uint32_t factor = std::clamp<uint32_t>(s_wantFactor.load(std::memory_order_relaxed), 1u, kMaxFactor);
            // In-between pictures need: the whole frame still buffered, a previous frame to move
            // from, and most of the matches believable (a camera cut matches the same models at
            // unrelated places).
            const uint32_t keyed = f.matched + f.rejected;
            const bool cut = keyed != 0u && f.rejected * 4u > keyed;
            // Falling behind the game (this flip reaches the GS thread long after the game issued
            // it): no extra pictures for half a second, so that the game's own rate is kept.
            const uint64_t issued = g_ps2FlipIssuedNs.exchange(0u, std::memory_order_relaxed);
            const uint64_t lag = issued != 0u && issued <= flipNs ? flipNs - issued : 0u;
            ++m_flipCount;
            if (m_shedding && factor > 1u && lag > m_periodNs + m_periodNs * 6u / 10u)
            {
                if (m_flipCount >= m_shedUntil)
                    ++m_statSheds;
                m_shedUntil = m_flipCount + 30u;
            }
            const bool shed = m_flipCount < m_shedUntil;
            m_statLagNs += lag;
            const bool usable = f.complete && !f.broken && m_prev.complete && factor > 1u && f.matched != 0u && !cut && !shed;
            const uint32_t passes = usable ? factor : 1u;
            static const bool s_log = std::getenv("PS2_FRAME_INTERP_LOG") != nullptr;
            if (s_log)
                std::fprintf(stderr, "[gs:interp] flip (match %.2f ms = pair %.2f + prims %.2f + recover %.2f): %zu prims in %zu objects, %zu ops, matched %u (+%u from neighbours) rejected %u, done ops %zu%s, prev complete %d -> %u pass(es)%s\n",
                             matchNs / 1e6, m_phaseNs[0] / 1e6, m_phaseNs[1] / 1e6, m_phaseNs[2] / 1e6, f.prims.size(), f.objects.size(), f.ops.size(), f.matched, f.recovered, f.rejected, f.doneOps, f.broken ? " (broken)" : "", m_prev.complete ? 1 : 0, passes, cut ? " (cut)" : "");

            // When to show them: the real frame `latency` after its flip, the in-between ones at
            // even steps before it; the period follows the game's flips.
            const uint64_t period = updatePeriod(flipNs);
            const uint64_t step = period / factor;
            const uint64_t latency = step * (factor - 1u) + m_renderNs + 3000000ull; // + the GPU's share
            uint64_t showReal = flipNs + latency;
            if (m_haveSchedule)
            {
                const uint64_t predicted = m_lastShowReal + period;
                const int64_t err = static_cast<int64_t>(showReal) - static_cast<int64_t>(predicted);
                if (std::llabs(err) < static_cast<int64_t>(period))
                    showReal = static_cast<uint64_t>(static_cast<int64_t>(predicted) + std::clamp<int64_t>(err / 8, -500000, 500000));
            }
            m_lastShowReal = showReal;
            m_haveSchedule = true;

            const uint64_t t0 = nowNs();
            for (uint32_t pass = 1; pass <= passes; ++pass)
            {
                const float t = static_cast<float>(pass) / static_cast<float>(passes);
                const bool real = pass == passes;
                replay(real ? f.doneOps : 0u, real ? 1.0f : t);
                s_nextDueNs.store(showReal - step * (passes - pass), std::memory_order_relaxed);
                m_inner->QueuePresentSnapshot(request);
                if (g_gsInterpPassHook)
                    g_gsInterpPassHook(m_inner.get(), pass, passes, real ? 1.0f : t);
                if (pass == 1u)
                {
                    // time from the flip until the first picture is drawn (smoothed)
                    const uint64_t first = nowNs() - flipNs;
                    m_renderNs = m_renderNs == 0u ? first : (m_renderNs * 7u + first) / 8u;
                }
            }
            const uint64_t t1 = nowNs();

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
            if (m_stats && m_statFlips >= 300u)
            {
                const double n = static_cast<double>(m_statFlips);
                std::fprintf(stderr, "[gs:interp] %ux: %.2f pictures per game frame; of %.0f primitives/frame %.1f%% matched, %.1f%% placed from neighbours, %.1f%% moved implausibly; matching %.2f ms/frame, drawing all passes %.2f ms/frame, game frame period %.2f ms, GS thread %.1f ms behind the game; frames without in-between pictures: %llu cut, %llu needed the renderer mid-frame (%llu such calls), %llu while behind (%llu times)\n",
                             factor, static_cast<double>(m_statPictures) / n, static_cast<double>(m_statPrims) / n,
                             m_statPrims ? 100.0 * m_statMatched / m_statPrims : 0.0, m_statPrims ? 100.0 * m_statRecovered / m_statPrims : 0.0,
                             m_statPrims ? 100.0 * m_statRejected / m_statPrims : 0.0, m_statMatchNs / 1e6 / n, m_statPassNs / 1e6 / n, period / 1e6,
                             m_statLagNs / 1e6 / n, static_cast<unsigned long long>(m_statCuts), static_cast<unsigned long long>(m_statBroken),
                             static_cast<unsigned long long>(m_statDrains), static_cast<unsigned long long>(m_statShedFrames),
                             static_cast<unsigned long long>(m_statSheds));
                m_statFlips = m_statPrims = m_statMatched = m_statRejected = m_statPictures = m_statPassNs = m_statCuts = m_statBroken = m_statDrains = 0;
                m_statLagNs = m_statShedFrames = m_statSheds = 0;
                m_statMatchNs = m_statRecovered = 0;
            }

            // This frame becomes "last frame" (its positions and objects; the ops are done).
            std::swap(m_prev, m_cur);
            m_prev.complete = true;
            m_cur.clear();
            m_runOpen = false;
            m_transferBegun = false;
            m_active = factor > 1u;
            m_cur.complete = m_active;
            if (!m_active)
                m_prev.clear();
        }

        // The game's frame period from its flips (smoothed; 1/60 s until known).
        uint64_t updatePeriod(uint64_t flipNs)
        {
            if (m_lastFlipNs != 0u)
            {
                const uint64_t d = flipNs - m_lastFlipNs;
                if (d > 4000000ull && d < 100000000ull)
                    m_periodNs = (m_periodNs * 15u + d) / 16u;
            }
            m_lastFlipNs = flipNs;
            return m_periodNs;
        }

        std::unique_ptr<GSRasterBackend> m_inner;
        SpinLock m_mutex;
        Frame m_cur, m_prev;
        bool m_active = false;
        bool m_runOpen = false;
        bool m_transferBegun = false; // a transfer was started since the last flip
        uint64_t m_kickBase = 0;
        struct Pair
        {
            uint32_t cur, prev;
        };
        struct Candidate
        {
            float distance;
            uint32_t cur, prev;
        };
        std::vector<Pair> m_pairs;
        std::vector<Candidate> m_candidates;
        std::vector<uint8_t> m_curTaken;
        std::vector<uint32_t> m_restCur, m_restPrev;
        struct TableEntry
        {
            uint64_t key = 0;
            uint32_t prev = 0;
            bool used = false;
        };
        std::vector<TableEntry> m_table;
        struct GridCell
        {
            float dx = 0.0f, dy = 0.0f, n = 0.0f;
        };
        std::vector<GridCell> m_grid;
        std::vector<uint32_t> m_gridQueue;
        struct FarPrim
        {
            uint32_t prim;
            uint8_t farMask;
        };
        std::vector<FarPrim> m_farPrims;
        uint64_t m_phaseNs[3]{};
        uint64_t m_flipCount = 0, m_shedUntil = 0, m_statLagNs = 0, m_statShedFrames = 0, m_statSheds = 0;
        bool m_shedding = !(std::getenv("PS2_FRAME_INTERP_SHED") && std::getenv("PS2_FRAME_INTERP_SHED")[0] == '0');
        std::vector<GSVertex> m_synth; // last-frame positions made up for unmatched primitives
        float m_maxMove = 96.0f;
        uint32_t m_dbgOnly = std::getenv("PS2_FRAME_INTERP_ONLY") ? static_cast<uint32_t>(std::atoi(std::getenv("PS2_FRAME_INTERP_ONLY"))) : 0u;

        uint64_t m_lastFlipNs = 0, m_periodNs = 16683333ull, m_renderNs = 0, m_lastShowReal = 0;
        bool m_haveSchedule = false;

        bool m_stats = false;
        uint64_t m_statFlips = 0, m_statPrims = 0, m_statMatched = 0, m_statRejected = 0, m_statPictures = 0, m_statPassNs = 0, m_statMatchNs = 0, m_statRecovered = 0,
                 m_statCuts = 0, m_statBroken = 0, m_statDrains = 0;
    };
}

std::unique_ptr<GSRasterBackend> ps2CreateInterpGsBackend(std::unique_ptr<GSRasterBackend> inner)
{
    if (!inner)
        return inner;
    return std::make_unique<GsInterpBackend>(std::move(inner));
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
