#include "fixture/function.h"
#include <nano/function/linprog.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_linprog)

UTEST_CASE(function)
{
    // see 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);

    const auto function = linprog_function_t{c, A};

    UTEST_CHECK_EQUAL(function.size(), 4);
    UTEST_CHECK_EQUAL(function.name(), "linprog[4D]");

    check_gradient(function);
    check_convexity(function);
    UTEST_CHECK_GREATER_EQUAL(function.strong_convexity(), 0.0);

    // TODO: check constraints
    // TODO: add other examples from the book
}

UTEST_END_MODULE()
