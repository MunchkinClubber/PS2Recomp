#version 450
// GS vertex: position in pixels (after XYOFFSET) with 1/16 precision, depth already scaled to
// 0..1 (z / 2^24), colour 0..255, S/T/Q (or U/V in 1/16 texels), fog 0..255.
layout(location = 0) in vec3 inPos;
layout(location = 1) in uvec4 inColor;
layout(location = 2) in vec3 inSTQ;
layout(location = 3) in float inFog;

layout(location = 0) noperspective out vec4 vColor;
layout(location = 1) flat out uvec4 vColorFlat;
layout(location = 2) noperspective out vec3 vSTQ;
layout(location = 3) noperspective out float vFog;
layout(location = 4) flat out float vZ;

void main()
{
    // The viewport is 4096x4096 at the origin, so these divisions are exact.
    gl_Position = vec4(inPos.x / 2048.0 - 1.0, inPos.y / 2048.0 - 1.0, inPos.z, 1.0);
    vColor = vec4(inColor);
    vColorFlat = inColor;
    vSTQ = inSTQ;
    vFog = inFog;
    vZ = inPos.z;
}
