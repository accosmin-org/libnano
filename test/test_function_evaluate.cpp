#include <fixture/function.h>

using namespace nano;

UTEST_BEGIN_MODULE()

UTEST_CASE(evaluate)
{
    for (const auto& function : function_t::make({4, 4, function_type::any}))
    {
        UTEST_REQUIRE(function);
        UTEST_CHECK_EQUAL(function->size(), 4);
        check_function(*function, function_config_t{});
    }
}

UTEST_END_MODULE()
