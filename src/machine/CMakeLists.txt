target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/machine/cluster.h
    ${CMAKE_SOURCE_DIR}/include/nano/machine/enums.h
    ${CMAKE_SOURCE_DIR}/include/nano/machine/params.h
    ${CMAKE_SOURCE_DIR}/include/nano/machine/result.h
    ${CMAKE_SOURCE_DIR}/include/nano/machine/stats.h
    ${CMAKE_SOURCE_DIR}/include/nano/machine/tune.h
    cluster.cpp
    params.cpp
    result.cpp
    stats.cpp
    tune.cpp)
