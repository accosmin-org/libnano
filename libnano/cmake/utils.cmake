# function to setup compilation flags for various targets
function(target_compile_configure target)
    target_compile_options(${target}
        PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Werror -pedantic>
        PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Werror -pedantic>
        PRIVATE $<$<CXX_COMPILER_ID:AppleClang>:-Wall -Wextra -Werror -pedantic>)

    target_compile_features(${target}
        PUBLIC cxx_std_14
        PRIVATE cxx_std_14)
endfunction()

# function to create a unit test application
function(make_test test libs)
    add_executable(${test} ${test}.cpp)
    target_compile_configure(${test})
    target_compile_definitions(${test} PRIVATE UTEST_WITH_EIGEN)
    target_include_directories(${test} SYSTEM PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../deps/utest>)
    target_link_libraries(${test} ${libs})
    add_test(${test} ${test})
endfunction()
