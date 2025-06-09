#pragma once

#include <nano/core/numeric.h>
#include <nano/function/util.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_random_x0(const function_t& function, const scalar_t scale = 1.0)
{
    return make_random_vector<scalar_t>(function.size(), -scale, +scale);
}

[[maybe_unused]] inline std::vector<vector_t> make_random_x0s(const function_t& function, const scalar_t scale = 1.0)
{
    std::vector<vector_t> vectors;
    vectors.emplace_back(make_random_x0(function, scale));
    vectors.emplace_back(make_full_vector<scalar_t>(function.size(), 0.0));

    return vectors;
}

[[maybe_unused]] inline auto check_gradient(const function_t& function, const int trials = 100,
                                            const scalar_t central_difference_epsilon = 1e-8,
                                            const scalar_t convex_subgradient_epsilon = 1e-10)
{
    const auto rfunction = function.clone();
    UTEST_REQUIRE(rfunction != nullptr);
    for (auto trial = 0; trial < trials; ++trial)
    {
        const auto x = make_random_x0(*rfunction);
        const auto z = make_random_x0(*rfunction);

        // (sub-)gradient approximation with centering difference
        UTEST_CHECK_LESS(grad_accuracy(*rfunction, x, central_difference_epsilon), central_difference_epsilon);

        // (sub-)gradient inequality for convex inequality
        if (rfunction->convex())
        {
            auto       gx      = vector_t{x.size()};
            const auto fz      = (*rfunction)(z);
            const auto fx      = (*rfunction)(x, gx);
            const auto epsilon = convex_subgradient_epsilon * 0.5 * (std::fabs(fx) + std::fabs(fz));
            UTEST_CHECK_GREATER_EQUAL(fz - fx + convex_subgradient_epsilon, gx.dot(z - x));
        }
    }
}

[[maybe_unused]] inline auto check_convexity(const function_t& function, const int trials = 100,
                                             const scalar_t epsilon = 1e-12)
{
    const auto rfunction = function.clone();
    UTEST_REQUIRE(rfunction != nullptr);
    for (auto trial = 0; trial < trials && function.convex(); ++trial)
    {
        const auto x0 = make_random_x0(*rfunction);
        const auto x1 = make_random_x0(*rfunction);
        UTEST_CHECK(is_convex(*rfunction, x0, x1, 20, epsilon));
    }
}
