target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/loss/cauchy.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/class.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/classnll.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/error.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/exponential.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/flatten.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/hinge.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/logistic.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/mae.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/mse.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/pinball.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/savage.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/squared_hinge.h
    ${CMAKE_SOURCE_DIR}/include/nano/loss/tangent.h
    pinball.cpp)
