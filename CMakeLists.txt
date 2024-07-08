cmake_minimum_required(VERSION 3.28)
project(Custosh LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

set(SOURCES
        src/scene.cpp
        src/scene.h
        src/utility.h
        src/custosh_except.h
        src/internal/windows_console_screen_buffer.h
        src/renderer.cu
        src/renderer.h
        src/internal/gpu_memory.h
        src/logger.cpp
        src/logger.h
)

add_library(custosh_lib STATIC ${SOURCES})

set_target_properties(custosh_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        CXX_STANDARD 20
)

target_include_directories(custosh_lib PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_include_directories(custosh_lib PUBLIC include)

add_executable(example example/main.cpp)

target_link_libraries(example custosh_lib)
