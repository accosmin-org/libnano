target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/local.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/space.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/step.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/callback.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/surrogate.h
    ${CMAKE_SOURCE_DIR}/include/nano/tuner/util.h
    local.cpp
    space.cpp
    surrogate.cpp
    util.cpp)
