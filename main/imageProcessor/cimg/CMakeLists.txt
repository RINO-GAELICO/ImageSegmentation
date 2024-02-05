cmake_minimum_required(VERSION 3.1)

project(cimg)

find_package(Threads REQUIRED)
find_package(X11 REQUIRED)

set(HEADERS
        src/images.h
)

set(SOURCES
        src/images.cpp
)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${X11_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC src)
target_include_directories(${PROJECT_NAME} PRIVATE CImg)
