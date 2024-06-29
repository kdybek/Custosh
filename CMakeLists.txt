cmake_minimum_required(VERSION 3.28)
project(Custosh LANGUAGES CXX CUDA)

add_executable(Custosh
        src/Main.cpp
        src/Utility.h
        src/Renderer.h
        src/Renderer.cpp
        src/Utility.h
        src/CustoshExcept.h
        src/Model.h
        src/WindowsConsoleScreenBuffer.h
)

set_target_properties(Custosh PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        CXX_STANDARD 20
)

find_package(CUDA REQUIRED)
target_include_directories(Custosh PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(Custosh PRIVATE ${CUDA_LIBRARIES})
