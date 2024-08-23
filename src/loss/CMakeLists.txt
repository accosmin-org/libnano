target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/loss/class.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/error.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/flatten.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/pinball.h
    pinball.cpp)
