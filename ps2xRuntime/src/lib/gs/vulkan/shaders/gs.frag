#version 450
// One shader for every GS draw: texture fetch with the GS wrap modes and bilinear filter,
// TFX combine, fog, alpha test, and the source side of the blend. The arithmetic mirrors
// GSCpuBackend (integer 0..255 values, truncations in the same places) so the two renderers
// agree to within the GPU's interpolation.
layout(location = 0) noperspective in vec4 vColor;
layout(location = 1) flat in uvec4 vColorFlat;
layout(location = 2) noperspective in vec3 vSTQ;
layout(location = 3) noperspective in float vFog;
layout(location = 4) flat in float vZ;

#ifdef DUAL_SRC
layout(location = 0, index = 0) out vec4 outColor;
layout(location = 0, index = 1) out vec4 outFactor;
#else
layout(location = 0) out vec4 outColor;
#endif

layout(set = 0, binding = 0) uniform usampler2D tex;

layout(push_constant) uniform PC
{
    uint flags;
    uint atest;   // atst | aref << 8 | ate << 16 | invert << 17
    uint wrap;    // wrapU | wrapV << 2
    uint regionU; // min | max << 16
    uint regionV;
    int texW;
    int texH;
    uint fogCol;  // r | g << 8 | b << 16
    int kS;       // blend: source coefficient of (A - B)
    int dS;       // blend: source coefficient of D
    uint blend;   // FIX | (C is FIX) << 8 | (C is Ad) << 9 | DATM << 10 | COLCLAMP << 11
    uint fbmsk;   // F_DSTREAD: FBMSK
    int kD;       // F_DSTREAD blend: destination coefficient of (A - B)
    int dD;       // F_DSTREAD blend: destination coefficient of D
    uint texa;    // ta0 | ta1 << 8 | aem << 16 (F_TEXA16 / F_TEXA24)
} pc;

layout(set = 1, binding = 0) uniform usampler2D dstTex; // copy of the target (F_DSTREAD)

const uint F_TME = 1u;
const uint F_FST_TRUNC = 2u;
const uint F_FST_ROUND = 4u;
const uint F_LINEAR = 8u;
const uint F_TCC = 16u;
const uint F_FGE = 128u;
const uint F_IIP = 256u;
const uint F_BLEND_EXACT = 512u;
const uint F_BLEND_SCALE = 1024u;
const uint F_FBA = 2048u;
const uint F_CT16 = 4096u;
const uint F_ZROUND = 8192u;
const uint F_ALPHA_FACTOR = 16384u; // no dual-source blending: the blend factor goes out as alpha
const uint F_SPLIT_LOW = 32768u;    // only pixels whose blend factor is at most one
const uint F_SPLIT_HIGH = 65536u;   // only pixels whose blend factor exceeds one
const uint F_OUT_FM1 = 131072u;     // output (factor - 1) in RGB
const uint F_BLEND_INT = 262144u;
const uint F_DSTREAD = 524288u;     // destination copy available: blend / DATE / FBMSK in the shader
const uint F_DATE = 1048576u;
const uint F_BLEND_DST = 2097152u;  // (with F_DSTREAD) blend with the full GS formula
const uint F_PABE = 4194304u;
const uint F_AD_HALF = 8388608u;    // Cs * Ad + Cd as two additive draws of Cs/256 * Ad
const uint F_TEXA24 = 16777216u;    // texture is a render target: apply TEXA as for PSMCT24
const uint F_TEXA16 = 33554432u;
const uint F_ZFLAT = 67108864u;
const uint F_BIAS_DOWN = 134217728u; // fixed-function blend rounds; bias the source so it truncates like the GS
const uint F_BIAS_UP = 268435456u;     // constant-depth primitive (sprite, line, point): exact Z from the vertex    // ... as for PSMCT16 (alpha stored as 0x80 / 0)   // source term of a blend with Cd * 1: exact integer |(kS*Cs*C >> 7) + dS*Cs|

int wrapCoord(int c, int size, uint mode, int mn, int mx)
{
    if (mode == 0u)
        return int(uint(c) & uint(size - 1));
    if (mode == 1u)
        return clamp(c, 0, size - 1);
    if (mode == 2u)
        return min(max(c, mn), mx);
    return int((uint(c) & uint(mn)) | uint(mx));
}

uvec4 fetchTexel(int u, int v)
{
    const uint wu = pc.wrap & 3u, wv = (pc.wrap >> 2) & 3u;
    u = wrapCoord(u, pc.texW, wu, int(pc.regionU & 0xFFFFu), int(pc.regionU >> 16));
    v = wrapCoord(v, pc.texH, wv, int(pc.regionV & 0xFFFFu), int(pc.regionV >> 16));
    const ivec2 size = textureSize(tex, 0);
    uvec4 t = texelFetch(tex, clamp(ivec2(u, v), ivec2(0), size - 1), 0);
    if ((pc.flags & (F_TEXA16 | F_TEXA24)) != 0u)
    {
        const bool rgbZero = t.r == 0u && t.g == 0u && t.b == 0u;
        const uint below = (((pc.texa >> 16) & 1u) != 0u && rgbZero) ? 0u : (pc.texa & 0xFFu);
        if ((pc.flags & F_TEXA16) != 0u && (t.a & 0x80u) != 0u)
            t.a = (pc.texa >> 8) & 0xFFu;
        else
            t.a = below;
    }
    return t;
}

uvec4 sampleTexture(float u, float v)
{
    if ((pc.flags & F_LINEAR) == 0u)
        return fetchTexel(int(u), int(v));
    precise float su = u - 0.5;
    precise float sv = v - 0.5;
    const int u0 = int(floor(su));
    const int v0 = int(floor(sv));
    precise float fx = su - float(u0);
    precise float fy = sv - float(v0);
    const uvec4 c00 = fetchTexel(u0, v0);
    const uvec4 c10 = fetchTexel(u0 + 1, v0);
    const uvec4 c01 = fetchTexel(u0, v0 + 1);
    const uvec4 c11 = fetchTexel(u0 + 1, v0 + 1);
    if (c00 == c10 && c00 == c01 && c00 == c11)
        return c00;
    precise vec4 a = vec4(c00), b = vec4(c10), c = vec4(c01), d = vec4(c11);
    precise vec4 top = a + (b - a) * fx;
    precise vec4 bottom = c + (d - c) * fx;
    precise vec4 r = top + (bottom - top) * fy;
    ivec4 n = ivec4(r);
    precise vec4 frac = r - vec4(n);
    n += ivec4(greaterThanEqual(frac, vec4(0.5)));
    return uvec4(clamp(n, ivec4(0), ivec4(255)));
}

bool alphaPasses(uint a)
{
    const uint atst = pc.atest & 7u;
    const uint aref = (pc.atest >> 8) & 0xFFu;
    switch (atst)
    {
    case 0u: return false;
    case 1u: return true;
    case 2u: return a < aref;
    case 3u: return a <= aref;
    case 4u: return a == aref;
    case 5u: return a >= aref;
    case 6u: return a > aref;
    default: return a != aref;
    }
}

void main()
{
    // Depth is z / 2^24: round to the integer Z the GS would store (as the CPU renderer does).
    if ((pc.flags & F_ZFLAT) != 0u)
        gl_FragDepth = vZ;
    else if ((pc.flags & F_ZROUND) != 0u)
        gl_FragDepth = round(gl_FragCoord.z * 16777216.0) / 16777216.0;
    else
        gl_FragDepth = gl_FragCoord.z;

    ivec4 col;
    if ((pc.flags & F_IIP) != 0u)
        col = clamp(ivec4(vColor), ivec4(0), ivec4(255));
    else
        col = ivec4(vColorFlat);

    if ((pc.flags & F_TME) != 0u)
    {
        float u, v;
        if ((pc.flags & F_FST_TRUNC) != 0u)
        {
            u = float(uint(vSTQ.x) & 0xFFFFu) / 16.0;
            v = float(uint(vSTQ.y) & 0xFFFFu) / 16.0;
        }
        else if ((pc.flags & F_FST_ROUND) != 0u)
        {
            u = float(clamp(int(vSTQ.x + 0.5), 0, 0xFFFF)) / 16.0;
            v = float(clamp(int(vSTQ.y + 0.5), 0, 0xFFFF)) / 16.0;
        }
        else
        {
            const float q = abs(vSTQ.z) > 1.0e-8 ? vSTQ.z : 1.0;
            precise float invQ = 1.0 / q;
            precise float pu = vSTQ.x * invQ * float(pc.texW);
            precise float pv = vSTQ.y * invQ * float(pc.texH);
            u = pu;
            v = pv;
        }
        const ivec4 t = ivec4(sampleTexture(u, v));
        const bool tcc = (pc.flags & F_TCC) != 0u;
        const uint tfx = (pc.flags >> 5) & 3u;
        ivec4 o;
        if (tfx == 0u)
        {
            o.rgb = clamp((t.rgb * col.rgb) >> 7, ivec3(0), ivec3(255));
            o.a = tcc ? clamp((t.a * col.a) >> 7, 0, 255) : col.a;
        }
        else if (tfx == 1u)
        {
            o.rgb = t.rgb;
            o.a = tcc ? t.a : col.a;
        }
        else if (tfx == 2u)
        {
            o.rgb = clamp(((t.rgb * col.rgb) >> 7) + col.a, ivec3(0), ivec3(255));
            o.a = tcc ? clamp(t.a + col.a, 0, 255) : col.a;
        }
        else
        {
            o.rgb = clamp(((t.rgb * col.rgb) >> 7) + col.a, ivec3(0), ivec3(255));
            o.a = tcc ? t.a : col.a;
        }
        col = o;
    }

    if ((pc.flags & F_FGE) != 0u)
    {
        const int fog = clamp(int(vFog), 0, 255);
        const ivec3 fc = ivec3(int(pc.fogCol & 0xFFu), int((pc.fogCol >> 8) & 0xFFu), int((pc.fogCol >> 16) & 0xFFu));
        col.rgb = ((fog * col.rgb) >> 8) + (((255 - fog) * fc) >> 8);
    }

    if (((pc.atest >> 16) & 1u) != 0u)
    {
        const bool invert = ((pc.atest >> 17) & 1u) != 0u;
        if (alphaPasses(uint(col.a)) == invert)
            discard;
    }

    if ((pc.flags & F_DSTREAD) != 0u)
    {
        // Everything the GS does with the destination, on a copy of the target taken before the
        // draw: DATE, the blend (A - B) * C >> 7 + D, COLCLAMP, FBA, FBMSK, the 16-bit store.
        const ivec4 d = ivec4(texelFetch(dstTex, ivec2(gl_FragCoord.xy), 0));
        if ((pc.flags & F_DATE) != 0u && (((d.a & 0x80) != 0) != (((pc.blend >> 10) & 1u) != 0u)))
            discard;
        const int cd = ((pc.blend & 0x100u) != 0u) ? int(pc.blend & 0xFFu) : (((pc.blend & 0x200u) != 0u) ? d.a : col.a);
        ivec3 v = col.rgb;
        if ((pc.flags & F_BLEND_DST) != 0u && !((pc.flags & F_PABE) != 0u && (col.a & 0x80) == 0))
        {
            v = (((pc.kS * v + pc.kD * d.rgb) * cd) >> 7) + pc.dS * v + pc.dD * d.rgb;
            v = clamp(v, ivec3(0), ivec3(255)); // COLCLAMP=0 (wrap) is not emulated, as in the CPU renderer
        }
        int a = col.a;
        if ((pc.flags & F_FBA) != 0u)
            a |= 0x80;
        uint px = uint(v.r) | (uint(v.g) << 8) | (uint(v.b) << 16) | (uint(a & 0xFF) << 24);
        const uint dpx = uint(d.r) | (uint(d.g) << 8) | (uint(d.b) << 16) | (uint(d.a) << 24);
        px = (px & ~pc.fbmsk) | (dpx & pc.fbmsk);
        if ((pc.flags & F_CT16) != 0u)
            px &= 0x80F8F8F8u;
        outColor = vec4(float(px & 0xFFu), float((px >> 8) & 0xFFu), float((px >> 16) & 0xFFu), float(px >> 24)) / 255.0;
#ifdef DUAL_SRC
        outFactor = vec4(0.0);
#endif
        return;
    }

    const int c = ((pc.blend & 0x100u) != 0u) ? int(pc.blend & 0xFFu) : col.a;
    if ((pc.flags & F_SPLIT_LOW) != 0u && c > 128)
        discard;
    if ((pc.flags & F_SPLIT_HIGH) != 0u && c <= 128)
        discard;
    int outA = col.a;
    if ((pc.flags & F_FBA) != 0u)
        outA |= 0x80;

    vec3 rgb;
    if ((pc.flags & F_OUT_FM1) != 0u)
        rgb = vec3(float(c) / 128.0 - 1.0);
    else if ((pc.flags & F_AD_HALF) != 0u)
        rgb = vec3(col.rgb) / 256.0;
    else if ((pc.flags & F_BLEND_SCALE) != 0u)
    {
        const float f = float(c) / 128.0;
        vec3 v = vec3(col.rgb) * abs(float(pc.kS) * f + float(pc.dS));
        if ((pc.flags & F_BIAS_DOWN) != 0u)
            v -= 0.49;
        else if ((pc.flags & F_BIAS_UP) != 0u)
            v += 0.49;
        rgb = v / 255.0;
    }
    else
    {
        ivec3 v = col.rgb;
        if ((pc.flags & F_BLEND_EXACT) != 0u)
            v = clamp(((pc.kS * v) * c >> 7) + pc.dS * v, ivec3(0), ivec3(255));
        else if ((pc.flags & F_BLEND_INT) != 0u)
            v = min(abs(((pc.kS * v) * c >> 7) + pc.dS * v), ivec3(255));
        if ((pc.flags & F_CT16) != 0u)
            v &= ivec3(0xF8);
        rgb = vec3(v) / 255.0;
    }
    if ((pc.flags & F_CT16) != 0u)
        outA &= 0x80;
    if ((pc.flags & F_ALPHA_FACTOR) != 0u)
        outColor = vec4(rgb, clamp(float(c) / 128.0, 0.0, 1.0));
    else
        outColor = vec4(rgb, float(outA) / 255.0);
#ifdef DUAL_SRC
    outFactor = vec4(0.0, 0.0, 0.0, float(c) / 128.0);
#endif
}
