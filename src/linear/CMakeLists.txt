target_sources(linear
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include/nano/linear/accumulator.h
        ${CMAKE_SOURCE_DIR}/include/nano/linear/function.h
        ${CMAKE_SOURCE_DIR}/include/nano/linear/result.h
        ${CMAKE_SOURCE_DIR}/include/nano/linear/util.h
        ${CMAKE_SOURCE_DIR}/include/nano/linear/lasso.h
        ${CMAKE_SOURCE_DIR}/include/nano/linear/ridge.h
        ${CMAKE_SOURCE_DIR}/include/nano/linear/ordinary.h
        ${CMAKE_SOURCE_DIR}/include/nano/linear/elastic_net.h

        accumulator.cpp
        function.cpp
        result.cpp
        util.cpp

        lasso.cpp
        ridge.cpp
        ordinary.cpp
        elastic_net.cpp)
