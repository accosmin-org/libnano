cmake_minimum_required(VERSION 3.5)
project(libnano VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include("${CMAKE_SOURCE_DIR}/cmake/utils.cmake")

##############################################################################
# setup versioning: library version & git hash commit

exec_program("git" ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "rev-parse HEAD"
    OUTPUT_VARIABLE PROJECT_GIT_COMMIT_HASH)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nano/version.h")

##############################################################################
# setup dependencies

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(NOT WIN32)
    find_package(Threads REQUIRED)
endif()

##############################################################################
# setup library

add_library(nano
    src/arch.cpp)

add_library(NANO::nano ALIAS nano)

target_include_directories(nano
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(NOT WIN32)
    target_link_libraries(nano
        PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endif()

target_compile_configure(nano)

##############################################################################
# setup command line utilities
# todo

##############################################################################
# setup unit tests

include(CTest)
enable_testing()

add_subdirectory(test)

##############################################################################
# setup installation

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/NANO)

install(TARGETS nano
    EXPORT nano-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(nano PROPERTIES EXPORT_NAME NANO)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(EXPORT nano-targets
    FILE NANOTargets.cmake
    NAMESPACE NANO::
    DESTINATION ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NANOConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/NANOConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NANOConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NANOConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NANOConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR})

export(EXPORT nano-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/NANOTargets.cmake
    NAMESPACE NANO::)

##############################################################################

message(STATUS "-----------------------------------------------------------------------------" "")
message(STATUS "SYSTEM:                        " "${CMAKE_SYSTEM_NAME}")
message(STATUS "PROCESSOR:                     " "${CMAKE_HOST_SYSTEM_PROCESSOR}")
message(STATUS "LINKER:                        " "${CMAKE_LINKER}")
message(STATUS "COMPILER:                      " "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "------------------------------------------------------------------------------" "")
message(STATUS "CXX FLAGS:                     " "${CMAKE_CXX_FLAGS}")
message(STATUS "CXX DEBUG FLAGS:               " "${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CXX RELEASE FLAGS:             " "${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CXX RELWITHDEBINFO FLAGS:      " "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CXX MINSIZEREL FLAGS:          " "${CMAKE_CXX_FLAGS_MINSIZEREL}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS:        " "${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "------------------------------------------------------------------------------" "")
message(STATUS "BUILD TYPE:                    " "${CMAKE_BUILD_TYPE}")
message(STATUS "------------------------------------------------------------------------------" "")
message(STATUS "ASAN                           " "${CMAKE_WITH_ASAN}")
message(STATUS "MSAN                           " "${CMAKE_WITH_MSAN}")
message(STATUS "TSAN                           " "${CMAKE_WITH_TSAN}")
message(STATUS "TESTS                          " "${CMAKE_WITH_TESTS}")
message(STATUS "CCACHE                         " "${CMAKE_WITH_CCACHE}")
message(STATUS "------------------------------------------------------------------------------" "")
