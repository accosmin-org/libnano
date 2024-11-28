target_sources(function PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/function/constraint.h
    ${CMAKE_SOURCE_DIR}/include/nano/function/lambda.h
    ${CMAKE_SOURCE_DIR}/include/nano/function/linear.h
    ${CMAKE_SOURCE_DIR}/include/nano/function/numeric.h
    ${CMAKE_SOURCE_DIR}/include/nano/function/penalty.h
    ${CMAKE_SOURCE_DIR}/include/nano/function/quadratic.h
    ${CMAKE_SOURCE_DIR}/include/nano/function/util.h
    constraint.cpp
    linear.cpp
    penalty.cpp
    quadratic.cpp
    util.cpp)

add_subdirectory(benchmark)
