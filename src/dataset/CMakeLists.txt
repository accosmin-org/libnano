target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/dataset/hash.h
    ${CMAKE_SOURCE_DIR}/include/nano/dataset/iterator.h
    ${CMAKE_SOURCE_DIR}/include/nano/dataset/scaling.h
    ${CMAKE_SOURCE_DIR}/include/nano/dataset/stats.h
    hash.cpp
    iterator.cpp
    stats.cpp)
