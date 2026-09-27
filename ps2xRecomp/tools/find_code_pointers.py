#!/usr/bin/env python3
"""Find code addresses reachable only through pointers (vtables, callback tables,
lui/addiu-built function pointers) or direct JAL calls that a Ghidra function map
does not cover.

Stripped retail games often have tiny virtual methods (getters/setters, empty
virtuals) that Ghidra never turns into functions because nothing calls them
directly. At runtime they are reached via JALR and fail with
"[guest-branch:missing-target]". Feed the output to PS2Recomp as entry points.

Usage:
  find_code_pointers.py GAME.ELF functions.csv [--toml]

  functions.csv  - the CSV exported by ExportPS2Functions.java
  --toml         - print an `entry_points = [...]` block to paste into [general]
"""
import bisect, csv, struct, sys


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    elf = open(sys.argv[1], 'rb').read()
    as_toml = '--toml' in sys.argv

    shoff = struct.unpack_from('<I', elf, 0x20)[0]
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', elf, 0x2e)
    raw = [struct.unpack_from('<IIIIII', elf, shoff + i * shentsize) for i in range(shnum)]
    stroff = raw[shstrndx][4]

    def name(n):
        return elf[stroff + n:elf.index(b'\0', stroff + n)].decode()

    secs = [(name(n), t, f, a, o, sz) for n, t, f, a, o, sz in raw]
    # executable PROGBITS, excluding VU microcode
    text = [(a, a + sz, o) for n, t, f, a, o, sz in secs
            if (f & 4) and t == 1 and a and not n.startswith('.vutext')]

    def in_text(x):
        return x % 4 == 0 and any(a <= x < b for a, b, _ in text)

    rows = [(int(r['Start'], 16), int(r['End'], 16), r['Name']) for r in csv.DictReader(open(sys.argv[2]))]
    starts = {s for s, _, _ in rows}

    found = {}
    # 1) 32-bit words in non-executable PROGBITS sections (.data, .rodata, .sdata, ...)
    for n, t, f, a, o, sz in secs:
        if t != 1 or (f & 4) or not a:
            continue
        for off in range(0, sz - 3, 4):
            v = struct.unpack_from('<I', elf, o + off)[0]
            if in_text(v) and v not in starts:
                found.setdefault(v, set()).add('data:' + n)
    # 2) lui + addiu/ori pairs in code building a code address
    for a, b, o in text:
        last = {}
        for pc in range(a, b, 4):
            ins = struct.unpack_from('<I', elf, o + pc - a)[0]
            op, rs, rt, imm = ins >> 26, (ins >> 21) & 31, (ins >> 16) & 31, ins & 0xffff
            if op == 0x03:  # JAL: direct call targets Ghidra never turned into functions
                v = ((pc + 4) & 0xf0000000) | ((ins & 0x03ffffff) << 2)
                if in_text(v) and v not in starts:
                    found.setdefault(v, set()).add('jal')
            if op == 0x0f:
                last[rt] = (imm << 16, pc)
            elif op in (0x09, 0x0d) and rs in last and pc - last[rs][1] <= 32:
                lo = imm - 0x10000 if (op == 0x09 and imm & 0x8000) else imm
                v = (last[rs][0] + lo) & 0xffffffff
                if in_text(v) and v not in starts:
                    found.setdefault(v, set()).add('code')

    def word(x):
        for a, b, o in text:
            if a <= x < b:
                return struct.unpack_from('<I', elf, o + x - a)[0]
        return None

    result = sorted(v for v in found if word(v) != 0)  # skip pointers at nop padding
    if as_toml:
        print('entry_points = [')
        for v in result:
            print(f'  "entry_{v:08x}@0x{v:x}",')
        print(']')
    else:
        for v in result:
            print(f'0x{v:x}\t{",".join(sorted(found[v]))}')
    print(f'# {len(result)} untracked code pointer targets', file=sys.stderr)


if __name__ == '__main__':
    main()
