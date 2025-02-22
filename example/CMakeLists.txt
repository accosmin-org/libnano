cmake_minimum_required(VERSION 3.5)
project(example VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include("${CMAKE_SOURCE_DIR}/../cmake/utils.cmake")

find_package(NANO 0.0.1 REQUIRED COMPONENTS core function solver)

include(CTest)
enable_testing()

function(make_example program)
    add_executable(${program} src/${program}.cpp)
    target_link_libraries(${program} PRIVATE ${ARGN})
    add_test(check_${program} ${program})
    copy_runtime_dlls(${program})
endfunction()

make_example(tensor NANO::core)
make_example(linprog NANO::solver)
make_example(quadprog NANO::solver)
make_example(nonlinear NANO::solver)
make_example(constrained NANO::solver)
