cmake_minimum_required(VERSION 3.5)
project(example VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(example src/main)

find_package(NANO 1.0 REQUIRED)
target_link_libraries(example NANO::NANO)

