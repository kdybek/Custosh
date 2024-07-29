cmake_minimum_required(VERSION 3.28)
project(Custosh LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

include_directories(include)

set(SOURCES
        src/scene.cpp
        src/scene.h
        src/utility.h
        src/custosh_except.h
        src/renderer.cu
        src/renderer.h
        src/logger.cpp
        src/logger.h
        src/internal/debug_macros.h
        src/internal/gpu_memory.h
        src/internal/windows_console_screen_buffer.h
)

add_library(custosh STATIC ${SOURCES})

set_target_properties(custosh PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(custosh PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_include_directories(custosh PUBLIC include)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(custosh PRIVATE CUSTOSH_DEBUG)
endif()

add_executable(demo examples/demo.cpp)

add_executable(test examples/test.cpp)

target_link_libraries(demo custosh)

target_link_libraries(test custosh)
