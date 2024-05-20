cmake_minimum_required(VERSION 3.22)
project(Custosh)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)

add_executable(Custosh
        src/main.cpp
        src/Utility.h
        src/Renderer.h
        src/Renderer.cpp
        src/Utility.h
        src/CustoshExcept.h
        src/Utility.cpp
)
