cmake_minimum_required(VERSION 3.28)
project(Custosh LANGUAGES CUDA CXX)

add_executable(Custosh
        src/Main.cpp
        src/Utility.cuh
        src/CustoshExcept.h
        src/WindowsConsoleScreenBuffer.h
        src/Renderer.cu
        src/Renderer.cuh
        src/Scene.h
        src/GPUMemoryManager.h
)

set_target_properties(Custosh PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        CXX_STANDARD 20
)

target_include_directories(Custosh PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
