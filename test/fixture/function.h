#include <utest/utest.h>
#include <nano/core/numeric.h>

using namespace nano;

[[maybe_unused]] static auto check_gradient(const function_t& function, int trials = 100, scalar_t epsilon_factor = 7.0)
{
    for (auto trial = 0; trial < trials; ++ trial)
    {
        const vector_t x = vector_t::Random(function.size());
        UTEST_CHECK_LESS(function.grad_accuracy(x), epsilon_factor * epsilon2<scalar_t>());
    }
}

[[maybe_unused]] static auto check_convexity(const function_t& function, int trials = 100)
{
    for (auto trial = 0; trial < trials && function.convex(); ++ trial)
    {
        const vector_t x0 = vector_t::Random(function.size());
        const vector_t x1 = vector_t::Random(function.size());
        UTEST_CHECK(function.is_convex(x0, x1, 20));
    }
}
