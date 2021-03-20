cmake_minimum_required(VERSION 3.10)
set(CMAKE_TOOLCHAIN_FILE ./vcpkg/scripts/buildsystems/vcpkg.cmake)

project(opencv_template)

include_directories(./vcpkg/installed/x64-windows/include)
link_directories(./vcpkg/installed/x64-windows/lib)


add_executable(write_text writetext.cpp)
