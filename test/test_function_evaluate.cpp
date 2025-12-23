#include <fixture/function.h>

using namespace nano;

UTEST_BEGIN_MODULE()

UTEST_CASE(evaluate)
{
    for (const auto& function : function_t::make({2, 4, function_type::any}))
    {
        UTEST_CHECK(function != nullptr);

        const auto funcname    = function->name();
        const auto constrained = funcname.find("#lasso") != string_t::npos ||    ///<
                                 funcname.find("#elasticnet") != string_t::npos; ///<

        const auto dims = function->size();
        UTEST_CHECK_LESS_EQUAL(dims, constrained ? (2 * dims) : dims);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        check_function(*function, function_config_t{});
    }
}

UTEST_END_MODULE()
