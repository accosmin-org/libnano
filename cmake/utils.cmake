# function to copy runtime DLLs for the given target (needed for shared library dependencies on MSVC)
# NB: from https://stackoverflow.com/questions/14089284
function(copy_runtime_dlls TARGET)
    get_property(already_applied TARGET "${TARGET}" PROPERTY _copy_runtime_dlls_applied)

    if (MSVC AND BUILD_SHARED_LIBS AND NOT already_applied)
        add_custom_command(
            TARGET "${TARGET}" POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -v -E copy
                "$<TARGET_RUNTIME_DLLS:${TARGET}>" "$<TARGET_FILE_DIR:${TARGET}>"
            COMMAND_EXPAND_LISTS
        )

        set_property(TARGET "${TARGET}" PROPERTY _copy_runtime_dlls_applied 1)
    endif ()
endfunction()

# function to configure the given library (e.g. alias, includes, installation)
function(make_lib lib)
    add_library(${lib})
    add_library(${CMAKE_PROJECT_NAME}::${lib} ALIAS ${lib})

    set_target_properties(${lib}
        PROPERTIES
            LANGUAGE CXX
            LINKER_LANGUAGE CXX)

    target_include_directories(${lib}
        PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src)

    # install the target and create export-set
    install(TARGETS ${lib}
        EXPORT ${lib}Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # generate and install export file
    install(EXPORT ${lib}Targets
        FILE ${lib}Targets.cmake
        NAMESPACE ${CMAKE_PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})
endfunction()

# function to create a unit test application
function(make_test test)
    add_executable(${test} ${test}.cpp)
    target_include_directories(${test}
        SYSTEM PRIVATE
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/test>
    )
    target_link_libraries(${test}
        PRIVATE ${ARGN}
    )

    if (NANO_ENABLE_LLVM_COV)
        # FIXME: Not working on Windows!
        add_test(
            NAME ${test}
            COMMAND bash -c "rm -f ${test}.profraw ${test}.profdata; \
                    LLVM_PROFILE_FILE=${test}.profraw ${CMAKE_CURRENT_BINARY_DIR}/${test}; \
                    llvm-profdata merge -sparse ${test}.profraw -o ${test}.profdata"
        )
    else()
        add_test(${test} ${test})
    endif()
endfunction()

# function to create an executable
function(make_app app)
    add_executable(${app} ${app}.cpp)
    target_include_directories(${app}
        PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
    target_link_libraries(${app}
        PRIVATE ${ARGN})
    install(TARGETS ${app}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

# function to install a project
function(install_project)
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

    install(FILES ${CMAKE_BINARY_DIR}/nano/version.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nano)

    # generate the version file for the config file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion)

    # create config file
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}
        NO_CHECK_REQUIRED_COMPONENTS_MACRO)

    # install config files
    install(
        FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})
endfunction()
