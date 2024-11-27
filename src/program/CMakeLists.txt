target_sources(program PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/program/benchmark.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/expected.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/stack.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/state.h
    benchmark.cpp
    state.cpp)
