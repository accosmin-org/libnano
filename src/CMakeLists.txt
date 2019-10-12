# create library

set(lsearch0_sources
    lsearch0.cpp
    lsearch0/linear.cpp
    lsearch0/constant.cpp
    lsearch0/cgdescent.cpp
    lsearch0/quadratic.cpp)

set(lsearchk_sources
    lsearchk.cpp
    lsearchk/fletcher.cpp
    lsearchk/backtrack.cpp
    lsearchk/cgdescent.cpp
    lsearchk/lemarechal.cpp
    lsearchk/morethuente.cpp)

set(solver_sources
    solver.cpp
    solver/gd.cpp
    solver/cgd.cpp
    solver/lbfgs.cpp
    solver/quasi.cpp)

set(loss_sources
    loss.cpp)

set(imclass_sources
    imclass.cpp
    imclass/cifar.cpp
    imclass/mnist.cpp)

set(util_sources
    util/table.cpp
    util/cmdline.cpp)

set(tabular_sources
    tabular.cpp
    tabular/iris.cpp
    tabular/wine.cpp
    tabular/adult.cpp
    tabular/abalone.cpp
    tabular/poker_hand.cpp
    tabular/forest_fires.cpp
    tabular/breast_cancer.cpp
    tabular/bank_marketing.cpp)

add_library(nano
    function.cpp
    ${loss_sources}
    ${util_sources}
    ${solver_sources}
    ${imclass_sources}
    ${tabular_sources}
    ${lsearch0_sources}
    ${lsearchk_sources})

add_library(NANO::nano ALIAS nano)
target_compile_configure(nano)

target_include_directories(nano
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(nano
    PUBLIC Eigen3::Eigen)

if(NOT WIN32)
    target_link_libraries(nano
        PUBLIC Threads::Threads
        PRIVATE Threads::Threads)
endif()

# install library
install(TARGETS nano
    EXPORT nano-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(nano PROPERTIES EXPORT_NAME NANO)
