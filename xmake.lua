-- xmake.lua
set_project("matmul-cpu")
set_languages("c++17")

add_rules("plugin.compile_commands.autoupdate", {outputdir = "build"})

 -- locate or download OpenBLAS
add_requires("openblas")

target("main")
    set_kind("binary")

    -- header search path
    add_includedirs("include")

    -- source files under src/
    add_files("src/*.cpp", "src/**/*.cpp")

    -- optimization and CPU-specific flags
    add_cxflags("-O3", "-march=native", "-fopenmp", {force = true})

    -- link flags: keep -fopenmp for the linker as well
    add_ldflags("-fopenmp", {force = true})

    -- link against OpenBLAS
    add_packages("openblas")