cmake_minimum_required(VERSION 3.5)
project(libnano VERSION 1.0.0 LANGUAGES CXX)

##################################################################################################
# setup project

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(NOT WIN32)
    find_package(Threads REQUIRED)
endif()

##################################################################################################
# setup library

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/NANO)
add_subdirectory(src)

##################################################################################################
# setup command line utilities

add_subdirectory(app)

##################################################################################################
# setup unit tests

include(CTest)
enable_testing()
add_subdirectory(test)

##################################################################################################
# setup installation

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

install(FILES ${CMAKE_BINARY_DIR}/nano/version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nano)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/../deps/json/include/nlohmann
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
