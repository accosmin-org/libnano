# function to setup compilation options
function(configure target)
    target_compile_options(${target}
        PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Werror -pedantic>
        PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Werror -pedantic>)

    target_compile_features(${target}
        PUBLIC cxx_std_14
        PRIVATE cxx_std_14)
endfunction()

# function to create a unit test application
function(make_test test)
    add_executable(${test} src/${test}.cpp)
    target_link_libraries(${test} NANO::nano)
    configure(${test})
    add_test(${test} ${test})
endfunction()
