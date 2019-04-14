# create library

set(lsearch0_sources
    lsearch0.cpp
    lsearch0/linear.cpp
    lsearch0/constant.cpp
    lsearch0/cgdescent.cpp
    lsearch0/quadratic.cpp)

set(lsearchk_sources
    lsearchk.cpp
    lsearchk/backtrack.cpp
    lsearchk/cgdescent.cpp
    lsearchk/lemarechal.cpp
    lsearchk/morethuente.cpp
    lsearchk/nocedalwright.cpp)

set(solver_sources
    solver.cpp
    solver/gd.cpp
    solver/cgd.cpp
    solver/lbfgs.cpp
    solver/quasi.cpp)

add_library(nano
    arch.cpp
    table.cpp
    logger.cpp
    cmdline.cpp
    function.cpp
    ${solver_sources}
    ${lsearch0_sources}
    ${lsearchk_sources})

add_library(NANO::nano ALIAS nano)
target_compile_configure(nano)

target_include_directories(nano
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../deps/json/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(NOT WIN32)
    target_link_libraries(nano
        PUBLIC ${CMAKE_THREAD_LIBS_INIT}
        PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endif()

# install library
install(TARGETS nano
    EXPORT nano-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(nano PROPERTIES EXPORT_NAME NANO)
