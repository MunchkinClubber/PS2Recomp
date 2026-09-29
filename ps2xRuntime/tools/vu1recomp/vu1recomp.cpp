// vu1recomp: translate captured VU1 microprogram images into C++ (see runtime/vu1_native.h).
//
// Input: one or more program dumps, each a sequence of records
//   u32 startPC, u32 hasImage, [16 KiB micro memory if hasImage]
// (written by the runtime's VU1 program dumper and the offline replay tool).
// Output: a .cpp with one function per distinct code image, registered at start-up.
//
// Uses the interpreter's own decoder (VU1Interpreter, through the VU1NativeAccess friend) so the
// timing information (latencies, register reads/writes, pipelines) is identical.

#include "runtime/ps2_vu1.h"

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

struct VU1NativeAccess
{
    using DP = VU1Interpreter::DecodedInstructionPair;
    using IU = VU1Interpreter::InstructionUsage;
    static DP decode(const VU1Interpreter &vu, const uint8_t *code, uint32_t pc) { return vu.decodeInstructionPair(code, pc); }
    static bool isPipe(const IU &u, int which)
    {
        switch (which)
        {
        case 0: return u.pipeline == VU1Interpreter::PipelineFdiv;
        case 1: return u.pipeline == VU1Interpreter::PipelineEfu;
        case 2: return u.pipeline == VU1Interpreter::PipelineXgkick;
        case 3: return u.pipeline == VU1Interpreter::PipelineBranch;
        }
        return false;
    }
};
using DP = VU1NativeAccess::DP;
using IU = VU1NativeAccess::IU;

namespace
{
    constexpr uint32_t kCodeSize = 0x4000u;
    constexpr uint32_t kPcMask = 0x3FFFu;

    std::string fmt(const char *f, ...)
    {
        char buf[1024];
        va_list ap;
        va_start(ap, f);
        vsnprintf(buf, sizeof(buf), f, ap);
        va_end(ap);
        return buf;
    }

    uint64_t fnv(const uint8_t *p, size_t n, uint64_t h = 1469598103934665603ull)
    {
        for (size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    }

    inline uint8_t laneBit(int c) { return static_cast<uint8_t>(8u >> c); }
    const char *laneName = "xyzw";

    int16_t imm11(uint32_t i) { return static_cast<int16_t>(static_cast<int32_t>(i << 21) >> 21); }
    uint8_t FTr(uint32_t i) { return (i >> 16) & 0x1F; }
    uint8_t FSr(uint32_t i) { return (i >> 11) & 0x1F; }
    uint8_t FDr(uint32_t i) { return (i >> 6) & 0x1F; }
    uint8_t VIT(uint32_t i) { return (i >> 16) & 0xF; }
    uint8_t VIS(uint32_t i) { return (i >> 11) & 0xF; }
    uint8_t VID(uint32_t i) { return (i >> 6) & 0xF; }
    uint8_t DEST(uint32_t i) { return (i >> 21) & 0xF; }

    struct Image
    {
        std::vector<uint8_t> code;
        std::set<uint32_t> entries;
    };

    enum class Flow
    {
        Next,      // falls through
        Branch,    // conditional / unconditional relative branch (B, BAL, IBxx)
        Jump,      // JR / JALR
        Unsupported
    };

    struct Lower
    {
        Flow flow = Flow::Next;
        bool unconditional = false;
        uint32_t target = 0; // for Branch
        bool isBal = false;
        std::string code;        // executes the lower op (reads old register values)
        std::string condition;   // for conditional branches: C++ bool expression
        std::string jumpTarget;  // for Jump: C++ expression of the target pc
        bool readsQ = false, readsP = false, readsClip = false, unsupported = false;
        // VF write done by the lower op (applied through temporaries so the upper/lower rules hold)
        uint8_t vfReg = 0, vfLanes = 0;
        std::string vfValue[4];
        uint8_t viReg = 0;        // VI written (value computed into `nvi`)
        std::string viValue;
    };

    struct Upper
    {
        std::string code;
        uint8_t vfReg = 0, vfLanes = 0;  // VF destination
        bool acc = false;                // ACC destination (lanes = vfLanes)
        bool readsQ = false, readsI = false;
        bool clip = false;
        std::string clipExpr;
        bool unsupported = false;
    };

    std::string vs(uint8_t r, int c) { return fmt("N(vf[%u][%d])", r, c); }

    // flags: produce MAC/status for FMAC ops (images that read them).
    Upper translateUpper(uint32_t up, bool flags)
    {
        Upper u;
        const uint8_t op = up & 0x3F;
        const uint8_t dest = DEST(up), fs = FSr(up), ft = FTr(up), fd = FDr(up);
        const uint8_t spc = (up & 3) | ((up >> 4) & 0x7C);
        const bool opmsub = op == 0x2E;
        const bool opmula = op >= 0x3C && spc == 0x2E;
        auto lanes = [&](auto body)
        {
            bool fmac = false;
            std::string code;
            for (int c = 0; c < 4; ++c)
            {
                if (!(dest & laneBit(c)))
                    continue;
                std::string e = body(c);
                const bool isFmacExpr = e.rfind("fAdd(", 0) == 0 || e.rfind("fSub(", 0) == 0 || e.rfind("fMul(", 0) == 0 ||
                                        e.rfind("fMadd(", 0) == 0 || e.rfind("fMsub(", 0) == 0 || ((opmsub || opmula) && e == "0.0f");
                if (!flags || !isFmacExpr)
                {
                    code += fmt("        const float u%d = %s;\n", c, e.c_str());
                    continue;
                }
                fmac = true;
                if (e == "0.0f")
                {
                    code += fmt("        float u%d = 0.0f;\n        const uint32_t f%d = exactFlags(u%d, 0.0L);\n", c, c, c);
                    if (opmsub)
                        code += fmt("        stk |= productFlags(N(vf[%u][3]), N(vf[%u][3]));\n", fs, ft);
                    continue;
                }
                const size_t paren = e.find('(');
                const std::string name = e.substr(0, paren), args = e.substr(paren + 1, e.size() - paren - 2);
                const bool product = name == "fMadd" || name == "fMsub";
                code += fmt("        uint32_t f%d = 0u;\n        const float u%d = %sF(%s, f%d%s);\n", c, c, name.c_str(), args.c_str(), c, product ? ", stk" : "");
            }
            if (fmac)
            {
                u.code += "        uint32_t stk = 0u;\n" + code;
                std::string mac = "0u", status = "0u";
                for (int c = 0; c < 4; ++c)
                    if (dest & laneBit(c))
                    {
                        mac += fmt(" | macBits(f%d, %uu)", c, laneBit(c));
                        status += fmt(" | f%d", c);
                    }
                u.code += fmt("        pushMacStatus(c, %s, %s, stk);\n", mac.c_str(), status.c_str());
            }
            else
                u.code += code;
        };
        auto bc = [&](int b) { return vs(ft, b); };
        const std::string Q = "N(s.q)", I = "N(s.i)";
        auto acc = [](int c) { return fmt("N(s.acc[%d])", c); };

        if (op <= 0x2F)
        {
            u.vfReg = fd;
            u.vfLanes = dest;
            switch (op)
            {
            case 0x00: case 0x01: case 0x02: case 0x03:
                lanes([&](int c) { return fmt("fAdd(%s, %s)", vs(fs, c).c_str(), bc(op & 3).c_str()); }); break;
            case 0x04: case 0x05: case 0x06: case 0x07:
                lanes([&](int c) { return fmt("fSub(%s, %s)", vs(fs, c).c_str(), bc(op & 3).c_str()); }); break;
            case 0x08: case 0x09: case 0x0A: case 0x0B:
                lanes([&](int c) { return fmt("fMadd(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), bc(op & 3).c_str()); }); break;
            case 0x0C: case 0x0D: case 0x0E: case 0x0F:
                lanes([&](int c) { return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), bc(op & 3).c_str()); }); break;
            case 0x10: case 0x11: case 0x12: case 0x13:
                lanes([&](int c) { return fmt("vmax(%s, %s)", vs(fs, c).c_str(), bc(op & 3).c_str()); }); break;
            case 0x14: case 0x15: case 0x16: case 0x17:
                lanes([&](int c) { return fmt("vmin(%s, %s)", vs(fs, c).c_str(), bc(op & 3).c_str()); }); break;
            case 0x18: case 0x19: case 0x1A: case 0x1B:
                lanes([&](int c) { return fmt("fMul(%s, %s)", vs(fs, c).c_str(), bc(op & 3).c_str()); }); break;
            case 0x1C: u.readsQ = true; lanes([&](int c) { return fmt("fMul(%s, %s)", vs(fs, c).c_str(), Q.c_str()); }); break;
            case 0x1D: u.readsI = true; lanes([&](int c) { return fmt("vmax(%s, %s)", vs(fs, c).c_str(), I.c_str()); }); break;
            case 0x1E: u.readsI = true; lanes([&](int c) { return fmt("fMul(%s, %s)", vs(fs, c).c_str(), I.c_str()); }); break;
            case 0x1F: u.readsI = true; lanes([&](int c) { return fmt("vmin(%s, %s)", vs(fs, c).c_str(), I.c_str()); }); break;
            case 0x20: u.readsQ = true; lanes([&](int c) { return fmt("fAdd(%s, %s)", vs(fs, c).c_str(), Q.c_str()); }); break;
            case 0x21: u.readsQ = true; lanes([&](int c) { return fmt("fMadd(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), Q.c_str()); }); break;
            case 0x22: u.readsI = true; lanes([&](int c) { return fmt("fAdd(%s, %s)", vs(fs, c).c_str(), I.c_str()); }); break;
            case 0x23: u.readsI = true; lanes([&](int c) { return fmt("fMadd(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), I.c_str()); }); break;
            case 0x24: u.readsQ = true; lanes([&](int c) { return fmt("fSub(%s, %s)", vs(fs, c).c_str(), Q.c_str()); }); break;
            case 0x25: u.readsQ = true; lanes([&](int c) { return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), Q.c_str()); }); break;
            case 0x26: u.readsI = true; lanes([&](int c) { return fmt("fSub(%s, %s)", vs(fs, c).c_str(), I.c_str()); }); break;
            case 0x27: u.readsI = true; lanes([&](int c) { return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), I.c_str()); }); break;
            case 0x28: lanes([&](int c) { return fmt("fAdd(%s, %s)", vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
            case 0x29: lanes([&](int c) { return fmt("fMadd(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
            case 0x2A: lanes([&](int c) { return fmt("fMul(%s, %s)", vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
            case 0x2B: lanes([&](int c) { return fmt("vmax(%s, %s)", vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
            case 0x2C: lanes([&](int c) { return fmt("fSub(%s, %s)", vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
            case 0x2D: lanes([&](int c) { return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
            case 0x2E:
            {
                static const int L[3] = {1, 2, 0}, R[3] = {2, 0, 1};
                lanes([&](int c) {
                    if (c == 3)
                        return std::string("0.0f");
                    return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, L[c]).c_str(), vs(ft, R[c]).c_str()); });
                break;
            }
            case 0x2F: lanes([&](int c) { return fmt("vmin(%s, %s)", vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
            }
            if (fd == 0)
                u.vfReg = 0; // vf0 writes are discarded (interpreter restores vf0 after each pair)
            return u;
        }
        if (op < 0x3C)
        {
            u.unsupported = true;
            return u;
        }
        const uint8_t sp = (up & 3) | ((up >> 4) & 0x7C);
        auto accOp = [&](auto body)
        {
            u.acc = true;
            u.vfLanes = dest;
            lanes(body);
        };
        switch (sp)
        {
        case 0x00: case 0x01: case 0x02: case 0x03:
            accOp([&](int c) { return fmt("fAdd(%s, %s)", vs(fs, c).c_str(), bc(sp & 3).c_str()); }); break;
        case 0x04: case 0x05: case 0x06: case 0x07:
            accOp([&](int c) { return fmt("fSub(%s, %s)", vs(fs, c).c_str(), bc(sp & 3).c_str()); }); break;
        case 0x08: case 0x09: case 0x0A: case 0x0B:
            accOp([&](int c) { return fmt("fMadd(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), bc(sp & 3).c_str()); }); break;
        case 0x0C: case 0x0D: case 0x0E: case 0x0F:
            accOp([&](int c) { return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), bc(sp & 3).c_str()); }); break;
        case 0x10: case 0x11: case 0x12: case 0x13:
        {
            static const char *div[4] = {"", " / 16.0f", " / 4096.0f", " / 32768.0f"};
            u.vfReg = ft;
            u.vfLanes = dest;
            lanes([&](int c) { return fmt("static_cast<float>(static_cast<int32_t>(fbits(vf[%u][%d])))%s", fs, c, div[sp & 3]); });
            break;
        }
        case 0x14: case 0x15: case 0x16: case 0x17:
        {
            static const char *scale[4] = {"1.0f", "16.0f", "4096.0f", "32768.0f"};
            u.vfReg = ft;
            u.vfLanes = dest;
            lanes([&](int c) { return fmt("bitsf(static_cast<uint32_t>(floatToInt(%s, %s)))", vs(fs, c).c_str(), scale[sp & 3]); });
            break;
        }
        case 0x18: case 0x19: case 0x1A: case 0x1B:
            accOp([&](int c) { return fmt("fMul(%s, %s)", vs(fs, c).c_str(), bc(sp & 3).c_str()); }); break;
        case 0x1C: u.readsQ = true; accOp([&](int c) { return fmt("fMul(%s, %s)", vs(fs, c).c_str(), Q.c_str()); }); break;
        case 0x1D:
            u.vfReg = ft;
            u.vfLanes = dest;
            lanes([&](int c) { return fmt("std::fabs(%s)", vs(fs, c).c_str()); });
            break;
        case 0x1E: u.readsI = true; accOp([&](int c) { return fmt("fMul(%s, %s)", vs(fs, c).c_str(), I.c_str()); }); break;
        case 0x1F:
            u.clip = true;
            u.clipExpr = fmt("clipBits(vf[%u], vf[%u][3])", fs, ft);
            break;
        case 0x20: u.readsQ = true; accOp([&](int c) { return fmt("fAdd(%s, %s)", vs(fs, c).c_str(), Q.c_str()); }); break;
        case 0x21: u.readsQ = true; accOp([&](int c) { return fmt("fMadd(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), Q.c_str()); }); break;
        case 0x22: u.readsI = true; accOp([&](int c) { return fmt("fAdd(%s, %s)", vs(fs, c).c_str(), I.c_str()); }); break;
        case 0x23: u.readsI = true; accOp([&](int c) { return fmt("fMadd(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), I.c_str()); }); break;
        case 0x24: u.readsQ = true; accOp([&](int c) { return fmt("fSub(%s, %s)", vs(fs, c).c_str(), Q.c_str()); }); break;
        case 0x25: u.readsQ = true; accOp([&](int c) { return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), Q.c_str()); }); break;
        case 0x26: u.readsI = true; accOp([&](int c) { return fmt("fSub(%s, %s)", vs(fs, c).c_str(), I.c_str()); }); break;
        case 0x27: u.readsI = true; accOp([&](int c) { return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), I.c_str()); }); break;
        case 0x28: accOp([&](int c) { return fmt("fAdd(%s, %s)", vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
        case 0x29: accOp([&](int c) { return fmt("fMadd(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
        case 0x2A: accOp([&](int c) { return fmt("fMul(%s, %s)", vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
        case 0x2C: accOp([&](int c) { return fmt("fSub(%s, %s)", vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
        case 0x2D: accOp([&](int c) { return fmt("fMsub(%s, %s, %s)", acc(c).c_str(), vs(fs, c).c_str(), vs(ft, c).c_str()); }); break;
        case 0x2E:
        {
            static const int L[3] = {1, 2, 0}, R[3] = {2, 0, 1};
            accOp([&](int c) {
                if (c == 3)
                    return std::string("0.0f");
                return fmt("fMul(%s, %s)", vs(fs, L[c]).c_str(), vs(ft, R[c]).c_str()); });
            break;
        }
        case 0x2F:
        case 0x30:
            break; // NOP
        default:
            u.unsupported = true;
            break;
        }
        if (!u.acc && u.vfReg == 0)
            u.vfLanes = 0;
        return u;
    }

    std::string addrExpr(const std::string &viExpr, int imm)
    {
        return fmt("((static_cast<uint32_t>(static_cast<int32_t>(%s + (%d))) * 16u) & 0x3FF0u)", viExpr.c_str(), imm);
    }
    std::string addrU16(const std::string &viExpr)
    {
        return fmt("((static_cast<uint32_t>(static_cast<uint16_t>(%s)) * 16u) & 0x3FF0u)", viExpr.c_str());
    }

    Lower translateLower(uint32_t lo, uint32_t pc)
    {
        Lower l;
        if (lo == 0u || lo == 0x8000033Cu)
            return l;
        const uint8_t opHi = (lo >> 25) & 0x7F;
        const uint8_t dest = DEST(lo);
        const int16_t imm = imm11(lo);
        const uint32_t branchTarget = (pc + 8u + static_cast<uint32_t>(static_cast<int32_t>(imm) * 8)) & kPcMask;
        auto loadVf = [&](uint8_t reg, const std::string &addr)
        {
            l.code += fmt("        const uint32_t la = %s;\n", addr.c_str());
            l.vfReg = reg;
            l.vfLanes = dest;
            for (int c = 0; c < 4; ++c)
                if (dest & laneBit(c))
                    l.vfValue[c] = fmt("ldf(m, la + %d)", c * 4);
        };
        auto storeVf = [&](uint8_t reg, const std::string &addr)
        {
            l.code += fmt("        { const uint32_t w[4] = {fbits(vf[%u][0]), fbits(vf[%u][1]), fbits(vf[%u][2]), fbits(vf[%u][3])}; storeQ(c, %s, w, %u); }\n",
                          reg, reg, reg, reg, addr.c_str(), dest);
        };
        auto viComp = [&]() { return (dest & 8) ? 0 : (dest & 4) ? 1 : (dest & 2) ? 2 : 3; };
        auto setVi = [&](uint8_t reg, const std::string &value)
        {
            if (reg == 0)
                return;
            l.viReg = reg;
            l.viValue = value;
        };
        auto branch = [&](const std::string &cond)
        {
            l.flow = Flow::Branch;
            l.target = branchTarget;
            l.condition = cond;
            l.unconditional = cond.empty();
        };

        switch (opHi)
        {
        case 0x00: loadVf(FTr(lo), addrExpr(fmt("vi[%u]", VIS(lo)), imm)); return l;
        case 0x01: storeVf(FSr(lo), addrExpr(fmt("vi[%u]", VIT(lo)), imm)); return l;
        case 0x04:
            setVi(VIT(lo), fmt("static_cast<int32_t>(static_cast<int16_t>(ldu(m, %s + %d) & 0xFFFFu))", addrExpr(fmt("vi[%u]", VIS(lo)), imm).c_str(), viComp() * 4));
            return l;
        case 0x05:
            l.code += fmt("        { const uint32_t v = static_cast<uint32_t>(static_cast<uint16_t>(vi[%u] & 0xFFFF)); const uint32_t w[4] = {v, v, v, v}; storeQ(c, %s, w, %u); }\n",
                          VIT(lo), addrExpr(fmt("vi[%u]", VIS(lo)), imm).c_str(), dest);
            return l;
        case 0x08:
        case 0x09:
        {
            const int16_t imm15 = static_cast<int16_t>((lo & 0x7FF) | ((lo >> 10) & 0x7800));
            setVi(VIT(lo), fmt("static_cast<int16_t>(vi[%u] %c (%d))", VIS(lo), opHi == 0x08 ? '+' : '-', imm15));
            return l;
        }
        case 0x10: l.readsClip = true; setVi(1, fmt("((s.clip & 0xFFFFFFu) == 0x%Xu) ? 1 : 0", lo & 0xFFFFFF)); return l;
        case 0x11: l.code += fmt("        fcset(c, 0x%Xu);\n", lo & 0xFFFFFF); return l;
        case 0x12: l.readsClip = true; setVi(1, fmt("((s.clip & 0x%Xu) != 0u) ? 1 : 0", lo & 0xFFFFFF)); return l;
        case 0x13: l.readsClip = true; setVi(1, fmt("((s.clip | 0x%Xu) == 0xFFFFFFu) ? 1 : 0", lo & 0xFFFFFF)); return l;
        case 0x1C: l.readsClip = true; setVi(VIT(lo), "static_cast<int32_t>(s.clip & 0x0FFFu)"); return l;
        case 0x14: case 0x15: case 0x16: case 0x17:
        {
            const uint32_t imm12 = (((lo >> 21) & 1u) << 11) | (lo & 0x7FFu);
            if (opHi == 0x15)
            {
                l.code += fmt("        fsset(c, 0x%Xu);\n", imm12);
                return l;
            }
            l.readsClip = true; // commits the flag pipeline
            if (opHi == 0x14)
                setVi(VIT(lo), fmt("((s.status & 0xFFFu) == 0x%Xu) ? 1 : 0", imm12));
            else if (opHi == 0x16)
                setVi(VIT(lo), fmt("static_cast<int32_t>((s.status & 0xFFFu) & 0x%Xu)", imm12));
            else
                setVi(VIT(lo), fmt("static_cast<int32_t>((s.status & 0xFFFu) | 0x%Xu)", imm12));
            return l;
        }
        case 0x18: case 0x1A: case 0x1B:
            l.readsClip = true;
            if (opHi == 0x18)
                setVi(VIT(lo), fmt("((s.mac & 0xFFFFu) == static_cast<uint32_t>(static_cast<uint16_t>(vi[%u]))) ? 1 : 0", VIS(lo)));
            else
                setVi(VIT(lo), fmt("static_cast<int32_t>(s.mac %c static_cast<uint32_t>(static_cast<uint16_t>(vi[%u])))", opHi == 0x1A ? '&' : '|', VIS(lo)));
            return l;
        case 0x20: branch(""); return l;
        case 0x21: branch(""); l.isBal = true; setVi(VIT(lo), fmt("%u", (pc + 16u) / 8u)); return l;
        case 0x24:
        case 0x25:
            l.flow = Flow::Jump;
            l.jumpTarget = fmt("((static_cast<uint32_t>(static_cast<uint16_t>(brVi(c, %u))) * 8u) & 0x3FFFu)", VIS(lo));
            if (opHi == 0x25)
                setVi(VIT(lo), fmt("%u", (pc + 16u) / 8u));
            return l;
        case 0x28: branch(fmt("static_cast<int16_t>(brVi(c, %u)) == static_cast<int16_t>(brVi(c, %u))", VIS(lo), VIT(lo))); return l;
        case 0x29: branch(fmt("static_cast<int16_t>(brVi(c, %u)) != static_cast<int16_t>(brVi(c, %u))", VIS(lo), VIT(lo))); return l;
        case 0x2C: branch(fmt("static_cast<int16_t>(brVi(c, %u)) < 0", VIS(lo))); return l;
        case 0x2D: branch(fmt("static_cast<int16_t>(brVi(c, %u)) > 0", VIS(lo))); return l;
        case 0x2E: branch(fmt("static_cast<int16_t>(brVi(c, %u)) <= 0", VIS(lo))); return l;
        case 0x2F: branch(fmt("static_cast<int16_t>(brVi(c, %u)) >= 0", VIS(lo))); return l;
        case 0x40: break;
        default: l.unsupported = true; return l;
        }

        const uint8_t funct = lo & 0x3F;
        const uint8_t vfT = FTr(lo), vfS = FSr(lo), viT = VIT(lo), viS = VIS(lo), viD = VID(lo);
        switch (funct)
        {
        case 0x30: setVi(viD, fmt("static_cast<int16_t>(vi[%u] + vi[%u])", viS, viT)); return l;
        case 0x31: setVi(viD, fmt("static_cast<int16_t>(vi[%u] - vi[%u])", viS, viT)); return l;
        case 0x32:
        {
            const int16_t imm5 = static_cast<int16_t>(static_cast<int32_t>(((lo >> 6) & 0x1F) << 27) >> 27);
            setVi(viT, fmt("static_cast<int16_t>(vi[%u] + (%d))", viS, imm5));
            return l;
        }
        case 0x34: setVi(viD, fmt("(vi[%u] & vi[%u])", viS, viT)); return l;
        case 0x35: setVi(viD, fmt("(vi[%u] | vi[%u])", viS, viT)); return l;
        default:
            if (funct < 0x3C)
            {
                l.unsupported = true;
                return l;
            }
        }
        const uint8_t sp = (lo & 3) | ((lo >> 4) & 0x7C);
        const int fsf = (lo >> 21) & 3, ftf = (lo >> 23) & 3;
        switch (sp)
        {
        case 0x30: // MOVE
            l.vfReg = vfT;
            l.vfLanes = dest;
            for (int c = 0; c < 4; ++c)
                if (dest & laneBit(c))
                    l.vfValue[c] = fmt("vf[%u][%d]", vfS, c);
            return l;
        case 0x31: // MR32
            l.vfReg = vfT;
            l.vfLanes = dest;
            for (int c = 0; c < 4; ++c)
                if (dest & laneBit(c))
                    l.vfValue[c] = fmt("vf[%u][%d]", vfS, (c + 1) & 3);
            return l;
        case 0x34: // LQI
            loadVf(vfT, addrU16(fmt("vi[%u]", viS)));
            setVi(viS, fmt("static_cast<int16_t>(vi[%u] + 1)", viS));
            return l;
        case 0x35: // SQI
            storeVf(vfS, addrU16(fmt("vi[%u]", viT)));
            setVi(viT, fmt("static_cast<int16_t>(vi[%u] + 1)", viT));
            return l;
        case 0x36: // LQD (pre-decrement)
            if (viS != 0)
            {
                l.code += fmt("        const int32_t dec = static_cast<int16_t>(vi[%u] - 1);\n", viS);
                loadVf(vfT, addrU16("dec"));
                setVi(viS, "dec");
            }
            else
                loadVf(vfT, addrU16("vi[0]"));
            return l;
        case 0x37: // SQD
            if (viT != 0)
            {
                l.code += fmt("        const int32_t dec = static_cast<int16_t>(vi[%u] - 1);\n", viT);
                storeVf(vfS, addrU16("dec"));
                setVi(viT, "dec");
            }
            else
                storeVf(vfS, addrU16("vi[0]"));
            return l;
        case 0x38: // DIV
            l.code += fmt("        divq(c, N(vf[%u][%d]), N(vf[%u][%d]));\n", vfS, fsf, vfT, ftf);
            return l;
        case 0x39: // SQRT
            l.code += fmt("        sqrtq(c, N(vf[%u][%d]));\n", vfT, ftf);
            return l;
        case 0x3A: // RSQRT
            l.code += fmt("        rsqrtq(c, N(vf[%u][%d]), N(vf[%u][%d]));\n", vfS, fsf, vfT, ftf);
            return l;
        case 0x3B: // WAITQ
            return l;
        case 0x3C: // MTIR
            setVi(viT, fmt("static_cast<int32_t>(static_cast<int16_t>(fbits(vf[%u][%d]) & 0xFFFFu))", vfS, fsf));
            return l;
        case 0x3D: // MFIR
            l.vfReg = vfT;
            l.vfLanes = dest;
            for (int c = 0; c < 4; ++c)
                if (dest & laneBit(c))
                    l.vfValue[c] = fmt("bitsf(static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(vi[%u] & 0xFFFF))))", viS);
            return l;
        case 0x3E: // ILWR
            setVi(viT, fmt("static_cast<int32_t>(static_cast<int16_t>(ldu(m, %s + %d) & 0xFFFFu))", addrU16(fmt("vi[%u]", viS)).c_str(), viComp() * 4));
            return l;
        case 0x3F: // ISWR
            l.code += fmt("        { const uint32_t v = static_cast<uint32_t>(static_cast<uint16_t>(vi[%u] & 0xFFFF)); const uint32_t w[4] = {v, v, v, v}; storeQ(c, %s, w, %u); }\n",
                          viT, addrU16(fmt("vi[%u]", viS)).c_str(), dest);
            return l;
        case 0x40: // RNEXT
            l.code += "        rnext(s);\n";
            l.vfReg = vfT;
            l.vfLanes = dest;
            for (int c = 0; c < 4; ++c)
                if (dest & laneBit(c))
                    l.vfValue[c] = "bitsf(s.r)";
            return l;
        case 0x41: // RGET
            l.vfReg = vfT;
            l.vfLanes = dest;
            for (int c = 0; c < 4; ++c)
                if (dest & laneBit(c))
                    l.vfValue[c] = "bitsf(s.r)";
            return l;
        case 0x42: l.code += fmt("        s.r = 0x3F800000u | (fbits(vf[%u][%d]) & 0x007FFFFFu);\n", vfS, fsf); return l;
        case 0x43: l.code += fmt("        s.r = 0x3F800000u | ((s.r ^ fbits(vf[%u][%d])) & 0x007FFFFFu);\n", vfS, fsf); return l;
        case 0x64: // MFP
            l.readsP = true;
            l.vfReg = vfT;
            l.vfLanes = dest;
            for (int c = 0; c < 4; ++c)
                if (dest & laneBit(c))
                    l.vfValue[c] = "s.p";
            return l;
        case 0x68: setVi(viT, "static_cast<int32_t>(s.top & 0x3FFu)"); return l;
        case 0x69: setVi(viT, "static_cast<int32_t>(s.itop & 0x3FFu)"); return l;
        case 0x6C: l.code += fmt("        xgStart(c, static_cast<uint32_t>(static_cast<uint16_t>(vi[%u])));\n", viS); return l;
        case 0x70: l.code += fmt("        { const float x = N(vf[%u][0]), y = N(vf[%u][1]), z = N(vf[%u][2]); queueP(c, x * x + y * y + z * z, 11u); }\n", vfS, vfS, vfS); return l;
        case 0x71: l.code += fmt("        { const float x = N(vf[%u][0]), y = N(vf[%u][1]), z = N(vf[%u][2]); const float sum = x * x + y * y + z * z; queueP(c, sum != 0.0f ? 1.0f / sum : sum, 18u); }\n", vfS, vfS, vfS); return l;
        case 0x72: l.code += fmt("        { const float x = N(vf[%u][0]), y = N(vf[%u][1]), z = N(vf[%u][2]); queueP(c, std::sqrt(x * x + y * y + z * z), 18u); }\n", vfS, vfS, vfS); return l;
        case 0x73: l.code += fmt("        { const float x = N(vf[%u][0]), y = N(vf[%u][1]), z = N(vf[%u][2]); const float len = std::sqrt(x * x + y * y + z * z); queueP(c, len != 0.0f ? 1.0f / len : len, 24u); }\n", vfS, vfS, vfS); return l;
        case 0x74: l.code += fmt("        { const float x = N(vf[%u][0]), y = N(vf[%u][1]); queueP(c, x != 0.0f ? eatan(y / x) : 0.0f, 54u); }\n", vfS, vfS); return l;
        case 0x75: l.code += fmt("        { const float x = N(vf[%u][0]), z = N(vf[%u][2]); queueP(c, x != 0.0f ? eatan(z / x) : 0.0f, 54u); }\n", vfS, vfS); return l;
        case 0x76: l.code += fmt("        { float sum = 0.0f; for (int k = 0; k < 4; ++k) sum += N(vf[%u][k]); queueP(c, sum, 12u); }\n", vfS); return l;
        case 0x77: l.code += fmt("        { const float v = N(vf[%u][%d]); float r = v; if (r >= 0.0f) { r = std::sqrt(r); if (r != 0.0f) r = 1.0f / r; } queueP(c, r, 18u); }\n", vfS, fsf); return l;
        case 0x78: l.code += fmt("        { const float v = N(vf[%u][%d]); queueP(c, v >= 0.0f ? std::sqrt(v) : v, 12u); }\n", vfS, fsf); return l;
        case 0x79: l.code += fmt("        queueP(c, esin(N(vf[%u][%d])), 29u);\n", vfS, fsf); return l;
        case 0x7A: l.code += fmt("        { const float v = N(vf[%u][%d]); queueP(c, v != 0.0f ? 1.0f / v : v, 12u); }\n", vfS, fsf); return l;
        case 0x7B: return l; // WAITP
        case 0x7C: l.code += fmt("        queueP(c, eatan(N(vf[%u][%d])), 54u);\n", vfS, fsf); return l;
        case 0x7D: l.code += fmt("        queueP(c, eexp(N(vf[%u][%d])), 44u);\n", vfS, fsf); return l;
        default:
            l.unsupported = true;
            return l;
        }
    }

    class Translator
    {
    public:
        Translator(const Image &img, std::string name) : m_img(img), m_name(std::move(name)) {}

        bool discover()
        {
            std::vector<uint32_t> work(m_img.entries.begin(), m_img.entries.end());
            for (uint32_t e : m_img.entries)
                m_leaders.insert(e);
            // Computed jumps: JR/JALR through a register loaded with IADDIU vi, vi0, target/8.
            // Collect those constants for registers used as jump bases (iterating as code is found).
            std::set<uint32_t> jumpRegs;
            std::set<uint32_t> scanned;
            for (;;)
            {
                exploreFrom(work);
                bool added = false;
                for (uint32_t pc : m_pairs)
                {
                    const DP d = decode(pc);
                    if (d.iBit)
                        continue;
                    const uint8_t opHi = (d.lower >> 25) & 0x7F;
                    if (opHi == 0x24 || opHi == 0x25)
                        jumpRegs.insert(VIS(d.lower));
                }
                for (uint32_t pc : m_pairs)
                {
                    const DP d = decode(pc);
                    if (d.iBit || ((d.lower >> 25) & 0x7F) != 0x08 || VIS(d.lower) != 0 || !jumpRegs.count(VIT(d.lower)))
                        continue;
                    const uint32_t imm = (d.lower & 0x7FF) | ((d.lower >> 10) & 0x7800);
                    const uint32_t target = imm * 8u;
                    if (target + 8u > kCodeSize || m_returnPoints.count(target))
                        continue;
                    m_returnPoints.insert(target);
                    m_leaders.insert(target);
                    work.push_back(target);
                    added = true;
                }
                if (!added)
                    break;
            }
            return true;
        }

        void exploreFrom(std::vector<uint32_t> &work)
        {
            while (!work.empty())
            {
                uint32_t pc = work.back();
                work.pop_back();
                while (pc + 8u <= kCodeSize && !m_pairs.count(pc))
                {
                    m_pairs.insert(pc);
                    const DP d = decode(pc);
                    Lower l = d.iBit ? Lower{} : translateLower(d.lower, pc);
                    if (d.eBit)
                    {
                        m_pairs.insert(pc + 8u); // delay slot; the program ends after it
                        break;
                    }
                    if (l.flow == Flow::Branch || l.flow == Flow::Jump)
                    {
                        m_pairs.insert(pc + 8u); // delay slot is emitted inline with the branch
                        if (l.flow == Flow::Branch)
                        {
                            m_leaders.insert(l.target);
                            work.push_back(l.target);
                        }
                        if (l.isBal || (!d.iBit && ((d.lower >> 25) & 0x7F) == 0x25))
                        {
                            m_returnPoints.insert(pc + 16u);
                            m_leaders.insert(pc + 16u);
                            work.push_back(pc + 16u);
                        }
                        if (l.flow == Flow::Branch && !l.unconditional)
                        {
                            m_leaders.insert(pc + 16u);
                            work.push_back(pc + 16u);
                        }
                        break;
                    }
                    pc += 8u;
                }
            }
        }

        // One function per basic block (keeps compile time and memory sane); each returns the next
        // PC, or kEnded / kHandoff. The program function dispatches on the PC.
        std::string emit()
        {
            std::string out;
            for (uint32_t pc : m_leaders)
                if (m_pairs.count(pc))
                    emitBlock(out, pc);
            out += fmt("static VU1NativeExit %s(VU1NativeCtx &c, uint32_t pc)\n{\n", m_name.c_str());
            out += "    for (;;)\n    {\n        uint32_t next;\n        switch (pc)\n        {\n";
            for (uint32_t pc : m_leaders)
                if (m_pairs.count(pc))
                    out += fmt("        case 0x%X: next = %s_%04X(c); break;\n", pc, m_name.c_str(), pc);
            out += "        default: return vu1n::handoff(c, pc);\n        }\n";
            out += "        if (next == kEnded)\n            return VU1NativeExit::Ended;\n";
            out += "        if (next == kHandoff)\n            return VU1NativeExit::Handoff;\n";
            out += "        pc = next;\n    }\n}\n";
            return out;
        }

        std::vector<std::pair<uint32_t, uint32_t>> spans() const
        {
            std::vector<std::pair<uint32_t, uint32_t>> r;
            for (uint32_t pc : m_pairs)
            {
                if (!r.empty() && r.back().second == pc)
                    r.back().second = pc + 8u;
                else
                    r.push_back({pc, pc + 8u});
            }
            return r;
        }
        uint64_t hash() const
        {
            uint64_t h = 1469598103934665603ull;
            for (auto [a, b] : spans())
                h = fnv(m_img.code.data() + a, b - a, h);
            return h;
        }
        size_t pairCount() const { return m_pairs.size(); }
        bool m_flags = false; // produce MAC/status flags (the image reads them)

        // MAC/status flags are not modelled, so an image whose code reads them stays interpreted
        // (a handoff mid-program would leave the interpreter with stale flags).
        bool readsMacOrStatus() const
        {
            for (uint32_t pc : m_pairs)
            {
                const DP d = decode(pc);
                if (d.iBit)
                    continue;
                const uint8_t opHi = (d.lower >> 25) & 0x7F;
                if (opHi == 0x14 || opHi == 0x16 || opHi == 0x17 || opHi == 0x18 || opHi == 0x1A || opHi == 0x1B)
                    return true;
            }
            return false;
        }

    private:
        const Image &m_img;
        std::string m_name;
        std::set<uint32_t> m_pairs, m_leaders, m_returnPoints;

        DP decode(uint32_t pc) const
        {
            static VU1Interpreter vu;
            return VU1NativeAccess::decode(vu, m_img.code.data(), pc);
        }

        std::string m_handoffArgs; // extra handoff() arguments while emitting a delay slot
        // Within a block: pair index from which a register lane is known to be ready (its write
        // landed or a stall on it already happened), so later reads need no stall check.
        std::map<std::string, int> m_known;
        int m_pairIndex = 0;
        bool knownReady(const std::string &key) const
        {
            auto it = m_known.find(key);
            return it != m_known.end() && it->second <= m_pairIndex;
        }

        void emitBlock(std::string &out, uint32_t start)
        {
            m_known.clear();
            m_pairIndex = 0;
            out += fmt("static uint32_t %s_%04X(VU1NativeCtx &c)\n{\n", m_name.c_str(), start);
            out += "    using namespace vu1n;\n";
            out += "    VU1State &s = *c.st;\n    float (*vf)[4] = s.vf;\n    int32_t *vi = s.vi;\n    uint8_t *m = c.mem;\n";
            out += "    (void)m;\n    (void)vf;\n    (void)vi;\n";
            emitBlockBody(out, start);
            out += "}\n\n";
        }

        void emitBlockBody(std::string &out, uint32_t start)
        {
            uint32_t pc = start;
            for (;;)
            {
                if (pc + 8u > kCodeSize)
                {
                    out += fmt("    return handoffRet(c, 0x%X);\n", pc & kPcMask);
                    return;
                }
                const DP d = decode(pc);
                const Lower l = d.iBit ? Lower{} : translateLower(d.lower, pc);
                if (d.eBit)
                {
                    emitPair(out, pc, d);
                    const DP ds = decode(pc + 8u);
                    m_handoffArgs = ", false, 0u, true";
                    emitPair(out, pc + 8u, ds);
                    m_handoffArgs.clear();
                    out += fmt("    s.pc = 0x%X;\n    finish(c);\n    return kEnded;\n", (pc + 16u) & kPcMask);
                    return;
                }
                if (l.flow == Flow::Branch || l.flow == Flow::Jump)
                {
                    emitBranch(out, pc, d, l);
                    return;
                }
                emitPair(out, pc, d);
                pc += 8u;
                if (m_leaders.count(pc))
                {
                    out += fmt("    return 0x%X;\n", pc);
                    return;
                }
            }
        }

        // Emits one instruction pair: stalls, commits, upper, lower, register writes, timing.
        // Returns nothing; for branch pairs the branch decision is emitted by emitBranch.
        void emitPair(std::string &out, uint32_t pc, const DP &d, const Lower *branchLower = nullptr)
        {
            const Upper u = translateUpper(d.upper, m_flags);
            const Lower l = d.iBit ? Lower{} : translateLower(d.lower, pc);
            out += fmt("    // %04X: %08X %08X\n", pc, d.upper, d.lower);
            if (u.unsupported || l.unsupported || d.upperUsage.reserved || d.lowerUsage.reserved)
            {
                out += fmt("    return handoffRet(c, 0x%X%s);\n", pc, m_handoffArgs.c_str());
                return;
            }
            if (d.dBit || d.tBit)
                out += fmt("    if (%s) return handoffRet(c, 0x%X%s);\n",
                           d.dBit && d.tBit ? "s.dBitEnabled || s.tBitEnabled" : d.dBit ? "s.dBitEnabled" : "s.tBitEnabled", pc, m_handoffArgs.c_str());
            out += "    {\n";
            // ---- stall
            std::vector<std::string> ready;
            auto vfReads = [&](const IU &usage)
            {
                for (uint32_t i = 0; i < usage.vfReadCount; ++i)
                {
                    const auto &a = usage.vfRead[i];
                    if (a.reg == 0)
                        continue;
                    for (int c = 0; c < 4; ++c)
                        if (a.lanes & laneBit(c))
                            ready.push_back(fmt("c.vfReady[%u][%d]", a.reg, c));
                }
                for (uint32_t r = 1; r < 16; ++r)
                    if (usage.viRead & (1u << r))
                        ready.push_back(fmt("c.viReady[%u]", r));
                for (int c = 0; c < 4; ++c)
                    if (usage.accRead & laneBit(c))
                        ready.push_back(fmt("c.accReady[%d]", c));
            };
            vfReads(d.upperUsage);
            vfReads(d.lowerUsage);
            std::sort(ready.begin(), ready.end());
            ready.erase(std::unique(ready.begin(), ready.end()), ready.end());
            for (const auto &r : ready)
            {
                if (knownReady(r))
                    continue;
                out += fmt("        stall(c, %s);\n", r.c_str());
                m_known[r] = m_pairIndex;
            }
            const bool fdivPipe = VU1NativeAccess::isPipe(d.lowerUsage, 0);
            const bool efuPipe = VU1NativeAccess::isPipe(d.lowerUsage, 1);
            const bool xgPipe = VU1NativeAccess::isPipe(d.lowerUsage, 2);
            if (fdivPipe || d.lowerUsage.waitQ)
                out += "        if (c.qPending) stall(c, c.qReady);\n";
            if (efuPipe)
                out += "        stall(c, c.efuResourceReady);\n";
            if (d.lowerUsage.waitP)
                out += "        stall(c, pAllReady(c));\n";
            if (xgPipe)
                out += "        xgWaitIdle(c);\n";
            // ---- commits visible at this cycle
            if (u.readsQ)
                out += "        commitQ(c);\n";
            if (l.readsP)
                out += "        commitP(c);\n";
            if (l.readsClip)
                out += "        commitClip(c);\n";
            // ---- upper
            out += u.code;
            if (u.clip)
                out += fmt("        queueClip(c, %s);\n", u.clipExpr.c_str());
            if (d.iBit)
                out += fmt("        s.i = N(bitsf(0x%08Xu));\n", d.lower);
            // ---- lower
            if (branchLower)
            {
                if (branchLower->flow == Flow::Branch && !branchLower->unconditional)
                    out += fmt("        taken = %s;\n", branchLower->condition.c_str());
                if (branchLower->flow == Flow::Jump)
                    out += fmt("        target = %s;\n", branchLower->jumpTarget.c_str());
            }
            out += l.code;
            if (l.vfReg != 0)
                for (int c = 0; c < 4; ++c)
                    if (l.vfLanes & laneBit(c))
                        out += fmt("        const float l%d = %s;\n", c, l.vfValue[c].c_str());
            if (l.viReg != 0)
                out += fmt("        const int32_t nvi = %s;\n", l.viValue.c_str());
            // ---- writes (upper after lower: lower saw the old values)
            const bool suppressLowerVf = d.suppressedLowerVf != 0 && l.vfReg == d.suppressedLowerVf;
            if (l.vfReg != 0 && !suppressLowerVf)
            {
                const uint32_t lat = d.lowerUsage.vfLatency ? d.lowerUsage.vfLatency : d.lowerUsage.latency;
                for (int c = 0; c < 4; ++c)
                    if (l.vfLanes & laneBit(c))
                    {
                        out += fmt("        vf[%u][%d] = l%d; c.vfReady[%u][%d] = c.cyc + %u;\n", l.vfReg, c, c, l.vfReg, c, lat);
                        m_known[fmt("c.vfReady[%u][%d]", l.vfReg, c)] = m_pairIndex + static_cast<int>(lat);
                    }
            }
            if (u.acc)
            {
                for (int c = 0; c < 4; ++c)
                    if (u.vfLanes & laneBit(c))
                    {
                        out += fmt("        s.acc[%d] = u%d; c.accReady[%d] = c.cyc + 1u;\n", c, c, c);
                        m_known[fmt("c.accReady[%d]", c)] = m_pairIndex + 1;
                    }
            }
            else if (u.vfReg != 0)
            {
                const uint32_t lat = d.upperUsage.vfLatency ? d.upperUsage.vfLatency : d.upperUsage.latency;
                for (int c = 0; c < 4; ++c)
                    if (u.vfLanes & laneBit(c))
                    {
                        out += fmt("        vf[%u][%d] = u%d; c.vfReady[%u][%d] = c.cyc + %u;\n", u.vfReg, c, c, u.vfReg, c, lat);
                        m_known[fmt("c.vfReady[%u][%d]", u.vfReg, c)] = m_pairIndex + static_cast<int>(lat);
                    }
            }
            const uint32_t viLat = d.lowerUsage.viLatency ? d.lowerUsage.viLatency : d.lowerUsage.latency;
            if (l.viReg != 0)
            {
                if (d.lowerUsage.delaysNextBranchRead)
                    out += fmt("        c.bkValid = true; c.bkReg = %u; c.bkVal = vi[%u];\n", l.viReg, l.viReg);
                else
                    out += "        c.bkValid = false;\n";
                out += fmt("        vi[%u] = nvi; c.viReady[%u] = c.cyc + %u;\n", l.viReg, l.viReg, viLat);
                m_known[fmt("c.viReady[%u]", l.viReg)] = m_pairIndex + static_cast<int>(viLat);
            }
            else
                out += "        c.bkValid = false;\n";
            out += "        ++c.cyc;\n    }\n";
            ++m_pairIndex;
        }

        void emitBranch(std::string &out, uint32_t pc, const DP &d, const Lower &l)
        {
            if (l.flow == Flow::Branch && !l.unconditional)
                out += "    bool taken = false;\n";
            if (l.flow == Flow::Jump)
                out += "    uint32_t target = 0;\n";
            emitPair(out, pc, d, &l);
            const DP ds = decode(pc + 8u);
            const Lower dl = ds.iBit ? Lower{} : translateLower(ds.lower, pc + 8u);
            const bool dsIsBranch = dl.flow != Flow::Next || ds.eBit;
            const std::string pendingTarget = l.flow == Flow::Jump ? "target" : fmt("0x%Xu", l.target);
            const std::string pendingTaken = (l.flow == Flow::Branch && !l.unconditional) ? "taken" : "true";
            if (dsIsBranch)
            {
                // Branch (or E bit) in a delay slot: let the interpreter take it from here.
                out += fmt("    return handoffRet(c, 0x%X, %s, %s);\n", pc + 8u, pendingTaken.c_str(), pendingTarget.c_str());
                return;
            }
            m_handoffArgs = ", " + pendingTaken + ", " + pendingTarget;
            emitPair(out, pc + 8u, ds);
            m_handoffArgs.clear();
            std::string next;
            if (l.flow == Flow::Jump)
                next = "target";
            else if (l.unconditional)
                next = fmt("0x%Xu", l.target);
            else
                next = fmt("(taken ? 0x%Xu : 0x%Xu)", l.target, (pc + 16u) & kPcMask);
            // Budget check (runaway loops end up in the interpreter, which stops them).
            out += fmt("    if (c.cyc >= c.budgetEnd)\n        return handoffRet(c, %s);\n", next.c_str());
            out += fmt("    return %s;\n", next.c_str());
        }
    };
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: vu1recomp out.cpp dump1.bin [dump2.bin ...]\n");
        return 1;
    }
    std::map<uint64_t, Image> images;
    for (int a = 2; a < argc; ++a)
    {
        FILE *f = std::fopen(argv[a], "rb");
        if (!f)
        {
            std::fprintf(stderr, "cannot open %s\n", argv[a]);
            return 1;
        }
        std::vector<uint8_t> cur(kCodeSize);
        uint64_t curHash = 0;
        bool have = false;
        uint32_t hdr[2];
        while (std::fread(hdr, 4, 2, f) == 2)
        {
            if (hdr[1])
            {
                if (std::fread(cur.data(), 1, kCodeSize, f) != kCodeSize)
                    break;
                curHash = fnv(cur.data(), kCodeSize);
                if (!images.count(curHash))
                    images[curHash].code = cur;
                have = true;
            }
            if (have)
                images[curHash].entries.insert(hdr[0] & kPcMask);
        }
        std::fclose(f);
    }

    std::string out;
    out += "// Generated by ps2xRuntime/tools/vu1recomp from captured VU1 micro memory. Do not edit.\n";
    out += "#include \"runtime/vu1_native.h\"\n\n#include <cmath>\n#include <cstring>\n\n";
    out += "namespace\n{\n";
    out += "    using namespace vu1n;\n";
    out += "    constexpr uint32_t kEnded = 0x10000u, kHandoff = 0x20000u;\n";
    out += "    inline uint32_t handoffRet(VU1NativeCtx &c, uint32_t pc, bool branch = false, uint32_t target = 0u, bool ebit = false)\n"
           "    {\n        handoff(c, pc, branch, target, ebit);\n        return kHandoff;\n    }\n";
    out += "    VU1N_INLINE float vmax(float a, float b) { return (a > b) ? a : b; }\n";
    out += "    VU1N_INLINE float vmin(float a, float b) { return (a < b) ? a : b; }\n";
    out += "    VU1N_INLINE float ldf(const uint8_t *m, uint32_t a) { float v; std::memcpy(&v, m + a, 4); return v; }\n";
    out += "    VU1N_INLINE uint32_t ldu(const uint8_t *m, uint32_t a) { uint32_t v; std::memcpy(&v, m + a, 4); return v; }\n";
    out += "    inline uint32_t clipBits(const float *v, float wf)\n    {\n"
           "        const uint32_t wb = fbits(wf);\n"
           "        const int32_t limit = (wb & 0x7F800000u) != 0u ? static_cast<int32_t>(wb & 0x7FFFFFFFu) : 0x007FFFFF;\n"
           "        auto ex = [limit](float value, uint32_t signMask) { return static_cast<int32_t>(fbits(value) ^ signMask) > limit; };\n"
           "        uint32_t f = 0u;\n"
           "        if (ex(v[0], 0u)) f |= 0x01u;\n        if (ex(v[0], 0x80000000u)) f |= 0x02u;\n"
           "        if (ex(v[1], 0u)) f |= 0x04u;\n        if (ex(v[1], 0x80000000u)) f |= 0x08u;\n"
           "        if (ex(v[2], 0u)) f |= 0x10u;\n        if (ex(v[2], 0x80000000u)) f |= 0x20u;\n"
           "        return f;\n    }\n";
    out += "    inline void divq(VU1NativeCtx &c, float num, float den)\n    {\n"
           "        uint32_t di = 0u;\n        float r;\n"
           "        if (den == 0.0f)\n        {\n            di = num == 0.0f ? 0x10u : 0x20u;\n"
           "            r = std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();\n        }\n"
           "        else\n            r = num / den;\n        queueQ(c, r, 7u, di);\n    }\n";
    out += "    inline void sqrtq(VU1NativeCtx &c, float v) { queueQ(c, std::sqrt(std::fabs(v)), 7u, v < 0.0f ? 0x10u : 0u); }\n";
    out += "    inline void rsqrtq(VU1NativeCtx &c, float num, float rad)\n    {\n"
           "        const float den = std::sqrt(std::fabs(rad));\n        uint32_t di = rad < 0.0f ? 0x10u : 0u;\n        float r;\n"
           "        if (den != 0.0f)\n            r = num / den;\n        else\n        {\n            di = num == 0.0f ? 0x10u : 0x20u;\n"
           "            r = std::signbit(num) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();\n        }\n"
           "        queueQ(c, r, 13u, di);\n    }\n";
    out += "    inline void rnext(VU1State &s)\n    {\n"
           "        const uint32_t x = (s.r >> 4) & 1u, y = (s.r >> 22) & 1u;\n"
           "        s.r = (((s.r << 1) ^ x ^ y) & 0x007FFFFFu) | 0x3F800000u;\n    }\n\n";

    int index = 0;
    std::string registrations;
    for (auto &[hash, img] : images)
    {
        const std::string name = fmt("vu1prog_%d", index);
        Translator t(img, name);
        t.discover();
        t.m_flags = t.readsMacOrStatus();
        if (t.m_flags)
            std::fprintf(stderr, "image %016llx (%zu entries) reads MAC/status: producing flags\n",
                         static_cast<unsigned long long>(hash), img.entries.size());
        out += t.emit();
        out += fmt("\nconst uint16_t %s_spans[] = {", name.c_str());
        for (auto [a, b] : t.spans())
            out += fmt("0x%X, 0x%X, ", a, b);
        out += "0, 0};\n";
        out += fmt("const uint16_t %s_entries[] = {", name.c_str());
        for (uint32_t e : img.entries)
            out += fmt("0x%X, ", e);
        out += "0xFFFF};\n";
        out += fmt("const VU1NativeImage %s_image = {\"%s\", 0x%016llXull, %s_spans, %s_entries, %s};\n",
                   name.c_str(), name.c_str(), static_cast<unsigned long long>(t.hash()), name.c_str(), name.c_str(), name.c_str());
        out += fmt("const vu1n::Registrar %s_reg(&%s_image);\n\n", name.c_str(), name.c_str());
        std::fprintf(stderr, "%s: %zu pairs, %zu entries\n", name.c_str(), t.pairCount(), img.entries.size());
        ++index;
    }
    out += "}\n\n// Referenced by the runtime so the linker keeps this translation unit.\nint g_vu1NativeGeneratedImages = " + std::to_string(index) + ";\n";

    FILE *o = std::fopen(argv[1], "wb");
    std::fwrite(out.data(), 1, out.size(), o);
    std::fclose(o);
    return 0;
}
