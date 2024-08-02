cmake_minimum_required(VERSION 3.28)
project(Custosh LANGUAGES CUDA CXX)

if(NOT WIN32)
    message(FATAL_ERROR "Custosh can only be built on Windows.")
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# --- Custosh ---
set(CUSTOSH_SOURCES
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

add_library(custosh STATIC ${CUSTOSH_SOURCES})

set_target_properties(custosh PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(custosh
        PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        INTERFACE include
)

target_compile_definitions(custosh PRIVATE $<$<CONFIG:Debug>:CUSTOSH_DEBUG>)

# --- Examples ---
macro(add_example EXAMPLE_NAME)

    add_executable(${EXAMPLE_NAME} ${ARGN})

    target_link_options(${EXAMPLE_NAME} PRIVATE
            $<$<CONFIG:Debug>:/NODEFAULTLIB:libcmt.lib>
            $<$<CONFIG:Release>:/NODEFAULTLIB:libcmtd.lib>
            /NODEFAULTLIB:msvcrt.lib
            /NODEFAULTLIB:msvcrtd.lib
    )

    target_link_libraries(${EXAMPLE_NAME} custosh)

endmacro()

# --- Demo ---
add_example(demo examples/demo.cpp)

# --- Test ---
add_example(test examples/test.cpp)
