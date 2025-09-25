-- xmake.lua
set_project("matmul_demo")
set_languages("c++17")

add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})

 -- locate or download OpenBLAS
add_requires("openblas")

target("basic")
    set_kind("binary")

    -- header search path
    add_includedirs("include")

    -- source files under src/
    add_files("src/basic/*.cpp", "src/utils.cpp")

    -- optimization and CPU-specific flags
    add_cxflags("-O3", "-march=native", {force = true})

target("openblas")
    set_kind("binary")

    -- header search path
    add_includedirs("include")

    -- source files under src/
    add_files("src/optimize/openblas.cpp", "src/utils.cpp")

    -- optimization and CPU-specific flags
    add_cxflags("-O3", "-march=native", {force = true})

    -- link against OpenBLAS
    add_packages("openblas")

target("salykova")
    set_kind("binary")

    -- header search path
    add_includedirs("include")

    -- source files under src/
    add_files("src/optimize/salykova.cpp", "src/utils.cpp")

    -- optimization and CPU-specific flags
    add_cxflags("-O3", "-march=native", {force = true})