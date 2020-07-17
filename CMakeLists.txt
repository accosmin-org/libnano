cmake_minimum_required(VERSION 3.5)
project(libnano
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Eigen-based numerical optimization and machine learning utilities")

option(NANO_BUILD_TESTS   "Build unit tests" ON)
option(NANO_BUILD_CMD_APP "Build command line utilities and benchmarks" ON)
option(NANO_BUILD_QT5_APP "Build Qt5-based interface (if Qt5 libraries are installed)" OFF)

##################################################################################################
# setup project

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY YES)

include(GNUInstallDirs)
include("${CMAKE_SOURCE_DIR}/cmake/utils.cmake")

exec_program("git" ${CMAKE_CURRENT_SOURCE_DIR}
    ARGS "rev-parse HEAD"
    OUTPUT_VARIABLE PROJECT_GIT_COMMIT_HASH)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nano/version.h")

set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -DEIGEN_MPL2_ONLY")
set(CMAKE_CXX_FLAGS                     "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE -DEIGEN_DEFAULT_TO_ROW_MAJOR")

set(CMAKE_CXX_FLAGS_RELEASE             "${CMAKE_CXX_FLAGS_RELEASE} -DEIGEN_NO_DEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DEIGEN_NO_DEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL          "${CMAKE_CXX_FLAGS_MINSIZEREL} -DEIGEN_NO_DEBUG")

##################################################################################################
# setup dependencies

if(NOT WIN32)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
    find_package(Threads REQUIRED)
endif()

find_package(Eigen3 3.3 NO_MODULE REQUIRED)

##################################################################################################
# setup library

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/NANO)
add_subdirectory(src)

##################################################################################################
# setup command line utilities and benchmark

if(NANO_BUILD_CMD_APP)
    add_subdirectory(app)
endif()

##################################################################################################
# setup unit tests

if(NANO_BUILD_TESTS)
    set(MEMORYCHECK_COMMAND_OPTIONS
        "--tool=memcheck --leak-check=full --show-reachable=yes --num-callers=50 --error-exitcode=1")
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()

##################################################################################################
# setup installation

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(FILES ${CMAKE_BINARY_DIR}/nano/version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nano)

install(EXPORT nano-targets
    FILE NANOTargets.cmake
    NAMESPACE NANO::
    DESTINATION ${INSTALL_CONFIGDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NANOConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

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

##################################################################################################

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

##################################################################################################
