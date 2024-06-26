make_lib(NANO program
    ${CMAKE_SOURCE_DIR}/include/nano/program/constrained.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/constraint.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/equality.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/inequality.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/linear.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/quadratic.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/solver.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/stack.h
    ${CMAKE_SOURCE_DIR}/include/nano/program/state.h

    constrained.cpp
    linear.cpp
    quadratic.cpp
    solver.cpp
    state.cpp
)
