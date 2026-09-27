include(CheckIPOSupported)

# SSX3: whole-program optimisation (/GL + /LTCG) makes the final link of 60k+
# generated functions run out of memory. Set -DPS2X_ENABLE_LTO=OFF to disable it.
option(PS2X_ENABLE_LTO "Enable /GL + /LTCG whole-program optimisation" ON)
if(PS2X_ENABLE_LTO)
    set(PS2X_GL_FLAG /GL)
    set(PS2X_LTCG_FLAG /LTCG)
else()
    set(PS2X_GL_FLAG "")
    set(PS2X_LTCG_FLAG "")
endif()

check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

function(EnableFastReleaseMode TargetName)
    message("> Enabling optimization for: ${TargetName}")
    if(MSVC)
        target_compile_options(${TargetName} PRIVATE
            $<$<CONFIG:Release>:
                /O2 # speed
                /Ob2 # inline aggressively
                /Oi # intrinsics
                ${PS2X_GL_FLAG} # whole program opt (optional)
                /Gy # function-level linking
                /Gw # global data in COMDAT
                /GF # string pooling
                /Zc:inline # remove unreferenced inline
                /fp:fast # fast math (graphics friendly)
                /DNDEBUG
                /arch:AVX2 # Advanced Vector Extensions 2
                /GS- # Disable Buffer Security Check (faster)
                /Qspectre- # Disable Spectre mitigations (faster)
            >
        )

        if(TARGET ${TargetName})
            target_link_options(${TargetName} PRIVATE
                $<$<CONFIG:Release>:
                    ${PS2X_LTCG_FLAG} # link-time code generation (optional)
                    /OPT:REF # remove unreferenced
                    /OPT:ICF # fold identical COMDATs
                >
            )
        endif()
    endif()

    if(IPO_SUPPORTED AND PS2X_ENABLE_LTO)
        set_property(TARGET ${TargetName} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    elseif(PS2X_ENABLE_LTO)
        message(WARNING "Interprocedural optimization not supported: ${ipo_error}")
    endif()
endfunction()