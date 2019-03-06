# create library
add_library(nano
    arch.cpp
    table.cpp
    logger.cpp
    solver.cpp
    solver/gd.cpp
    solver/cgd.cpp
    solver/lbfgs.cpp
    solver/quasi.cpp
    lsearch/init.cpp
    lsearch/backtrack.cpp
    lsearch/cgdescent.cpp
    lsearch/lemarechal.cpp
    lsearch/morethuente.cpp
    lsearch/nocedalwright.cpp
    cmdline.cpp
    function.cpp)

add_library(NANO::nano ALIAS nano)
target_compile_configure(nano)

target_include_directories(nano
    PUBLIC
        $<INSTALL_INTERFACE:include>
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
