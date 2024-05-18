cmake_minimum_required(VERSION 3.22)
project(Custosh)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)

add_executable(Custosh
        main.cpp
        Matrix.h
)
