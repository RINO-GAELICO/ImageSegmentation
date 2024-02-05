cmake_minimum_required(VERSION 3.1)

project(MeanShiftParallel)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE "Release")

add_subdirectory(edison)
add_subdirectory(main)


