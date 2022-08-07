#include <nano/core/numeric.h>
#include <nano/function/util.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] static vector_t make_random_x0(const function_t& function, const scalar_t scale = 1.0)
{
    return make_random_tensor<scalar_t>(make_dims(function.size()), -scale, +scale).vector();
}

[[maybe_unused]] static std::vector<vector_t> make_random_x0s(const function_t& function, const scalar_t scale = 1.0)
{
    const auto dims = make_dims(function.size());

    std::vector<vector_t> vectors;
    vectors.emplace_back(make_random_tensor<scalar_t>(make_dims(function.size()), -scale, +scale).vector());

    // BUG: OSGA solver
    if (function.name() == "Exponential[4D]")
    {
        vectors.emplace_back(make_tensor<scalar_t>(dims, 0.9460835747689484, 0.3166827894775206, -0.0416191904634331,
                                                   -0.9001861362105115)
                                 .vector());
    }

    return vectors;
}

[[maybe_unused]] static auto check_gradient(const function_t& function, int trials = 100, scalar_t epsilon_factor = 5.0)
{
    for (auto trial = 0; trial < trials; ++trial)
    {
        const auto x = make_random_x0(function);
        UTEST_CHECK_LESS(grad_accuracy(function, x), epsilon_factor * epsilon2<scalar_t>());
    }
}

[[maybe_unused]] static auto check_convexity(const function_t& function, int trials = 100)
{
    for (auto trial = 0; trial < trials && function.convex(); ++trial)
    {
        const auto x0 = make_random_x0(function);
        const auto x1 = make_random_x0(function);
        UTEST_CHECK(is_convex(function, x0, x1, 20));
    }
}
