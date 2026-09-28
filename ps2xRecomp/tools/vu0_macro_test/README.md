Differential test for the VU0 macro-mode (COP2) translations.

gen.cpp links libps2_recomp_lib and emits the translator's C++ for ~300 COP2 upper/lower encodings
into cases.inc; test.cpp compiles those snippets against the runtime headers, runs them on random
register values and compares with a small reference model of the PS2 semantics.

    g++ -std=c++20 -I ps2xRecomp/include -I <fmt/toml includes> gen.cpp libps2_recomp_lib.a librabbitizer.a libfmt.a -o gen && ./gen
    g++ -std=c++20 -msse4.1 -I ps2xRuntime/include -I ps2xRuntime/src/lib -I ps2xRuntime/src/lib/Kernel -I <ps2xIOP include> test.cpp -o test && ./test

Known remaining differences: the w lane of VOPMULA/VOPMSUB when the dest mask includes w
(real code always uses .xyz).
