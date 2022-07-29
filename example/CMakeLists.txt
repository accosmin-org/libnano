cmake_minimum_required(VERSION 3.5)
project(example VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(info src/info.cpp)
add_executable(tensor src/tensor.cpp)
add_executable(minimize src/minimize.cpp)

find_package(NANO 0.0.1 REQUIRED)

target_link_libraries(info NANO::NANO)
target_link_libraries(tensor NANO::NANO)
target_link_libraries(minimize NANO::NANO)

include(CTest)
enable_testing()

add_test(tensor tensor)
add_test(minimize minimize)
