target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/accumulator.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/early_stopping.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/enums.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/function.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/model.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/result.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/sampler.h
    ${CMAKE_SOURCE_DIR}/include/nano/gboost/util.h
    accumulator.cpp
    early_stopping.cpp
    function.cpp
    model.cpp
    result.cpp
    sampler.cpp
    util.cpp)
