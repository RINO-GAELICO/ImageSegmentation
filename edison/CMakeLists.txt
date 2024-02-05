cmake_minimum_required(VERSION 3.1)

project(edison LANGUAGES C CXX CUDA)

set(HEADERS
        code/region_creator.h
        segm/ms.h
        segm/msImageProcessor.h
        segm/RAList.h
        segm/rlist.h
        segm/tdef.h
        code/likwid-stuff.h
)

set(SOURCES
        
        code/ms_filter_omp.cpp
        code/ms_filter_serial.cpp
        code/ms_filter_cuda.cu
        code/region_creator.cpp
        segm/ms.cpp
        segm/msImageProcessor.cpp
        segm/RAList.cpp
        segm/rlist.cpp
)

find_package(OpenMP REQUIRED)
find_package(OpenMP REQUIRED)
find_package(LIKWID QUIET)
if ( LIKWID_FOUND ) 
   message ( STATUS "Likwid: found")
else()
   message( STATUS  "Likwid: NOT found")
endif()

if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC code ${LIKWID_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE segm)
target_link_libraries(${PROJECT_NAME} ${LIKWID_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE  -DLIKWID_PERFMON)
# target_compile_definitions( ${PROJECT_NAME} PUBLIC code -DLIKWID_PERFMON)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -march=native  -fopenmp)



