cmake_minimum_required(VERSION 3.5)
project(example VERSION 0.0.1 LANGUAGES CXX)

add_executable(example src/main)
target_compile_features(example PRIVATE cxx_std_14)

find_package(NANO 1.0 REQUIRED)
target_link_libraries(example NANO::NANO)

