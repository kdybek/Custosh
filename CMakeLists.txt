cmake_minimum_required(VERSION 3.28)
project(Custosh LANGUAGES CUDA CXX)

add_executable(Custosh
        src/Main.cu
        src/Utility.cuh
        src/Renderer.cuh
        src/Renderer.cu
        src/CustoshExcept.h
        src/Mesh.h
        src/WindowsConsoleScreenBuffer.h
        src/TriangleRegistry.cuh
)

set_target_properties(Custosh PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        CXX_STANDARD 20
)

target_include_directories(Custosh PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
