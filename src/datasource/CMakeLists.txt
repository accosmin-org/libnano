target_sources(machine PRIVATE
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/csv.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/imclass_cifar.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/imclass_mnist.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/iterator.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/linear.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/mask.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/storage.h
    ${CMAKE_SOURCE_DIR}/include/nano/datasource/tabular.h
    imclass_cifar.cpp
    imclass_mnist.cpp
    linear.cpp
    mask.cpp
    tabular.cpp)
