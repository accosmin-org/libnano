#include <nano/core/numeric.h>
#include <nano/function/util.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] static auto check_gradient(const function_t& function, int trials = 100, scalar_t epsilon_factor = 5.0)
{
    for (auto trial = 0; trial < trials; ++trial)
    {
        const vector_t x = vector_t::Random(function.size());
        UTEST_CHECK_LESS(grad_accuracy(function, x), epsilon_factor * epsilon2<scalar_t>());
    }
}

[[maybe_unused]] static auto check_convexity(const function_t& function, int trials = 100)
{
    for (auto trial = 0; trial < trials && function.convex(); ++trial)
    {
        const vector_t x0 = vector_t::Random(function.size());
        const vector_t x1 = vector_t::Random(function.size());
        UTEST_CHECK(is_convex(function, x0, x1, 20));
    }
}

[[maybe_unused]] static vector_t make_random_x0(const function_t& function, const scalar_t scale = 1.0)
{
    return make_random_tensor<scalar_t>(make_dims(function.size()), -scale, +scale).vector();
}
