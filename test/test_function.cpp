#include <utest/utest.h>
#include <nano/numeric.h>
#include <nano/function.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_functions)

UTEST_CASE(evaluate)
{
    for (const auto& rfunction : get_functions(1, 4, std::regex(".+")))
    {
        const auto& function = *rfunction;
        std::cout << function.name() << std::endl;

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 1);

        const auto trials = size_t(100);
        for (size_t t = 0; t < trials; ++ t)
        {
            const vector_t x0 = vector_t::Random(dims);
            const vector_t x1 = vector_t::Random(dims);

            UTEST_CHECK_LESS(function.grad_accuracy(x0), 10 * epsilon2<scalar_t>());
            UTEST_CHECK(!function.is_convex() || function.is_convex(x0, x1, 20));
        }
    }
}

UTEST_END_MODULE()
