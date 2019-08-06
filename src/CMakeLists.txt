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

set(dataset_sources
    dataset.cpp
    dataset/iris.cpp
    dataset/wine.cpp
    dataset/adult.cpp
    dataset/tabular.cpp)

add_library(nano
    table.cpp
    cmdline.cpp
    function.cpp
    ${loss_sources}
    ${solver_sources}
    ${dataset_sources}
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

#target_link_libraries(nano
#    PUBLIC nlohmann_json::nholmann_json)

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
