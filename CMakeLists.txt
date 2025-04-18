cmake_minimum_required(VERSION 3.10)
project(neural-networks)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SOURCES "src/*.cpp" "tests/*.cpp")

add_executable(neural-networks ${SOURCES})
target_include_directories(neural-networks PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(neural-networks PRIVATE ${CMAKE_SOURCE_DIR}/external/eigen)
target_include_directories(neural-networks PRIVATE ${CMAKE_SOURCE_DIR}/external/EigenRand)
