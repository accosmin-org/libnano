#include <fixture/function.h>

using namespace nano;

UTEST_BEGIN_MODULE()

UTEST_CASE(evaluate)
{
    for (const auto& function : function_t::make({2, 4, function_type::any}))
    {
        UTEST_REQUIRE(function);
        UTEST_CHECK_LESS_EQUAL(function->size(), 4);
        UTEST_CHECK_GREATER_EQUAL(function->size(), 2);

        check_function(*function, function_config_t{});
    }
}

UTEST_END_MODULE()
