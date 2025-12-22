#include <fixture/function.h>

using namespace nano;

UTEST_BEGIN_MODULE()

UTEST_CASE(evaluate)
{
    for (const auto& rfunction : function_t::make({2, 4, function_type::any}))
    {
        const auto& function = *rfunction;

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 8);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        check_function(function, function_config_t{});
    }
}

UTEST_END_MODULE()
