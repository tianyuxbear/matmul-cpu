-- xmake.lua
set_project("matmul_demo")
set_languages("c++17")

add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})

target("basic")
    set_kind("binary")

    -- header search path
    add_includedirs("include")

    -- source files under src/
    add_files("src/basic/*.cpp", "src/utils.cpp")

    -- optimization and CPU-specific flags
    add_cxflags("-O3", "-march=native", {force = true})