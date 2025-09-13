#pragma once

#include <nano/core/numeric.h>
#include <nano/function/util.h>
#include <utest/utest.h>

using namespace nano;

template <class... targs>
[[maybe_unused]] inline auto make_function(const string_t& name, const targs&... args)
{
    auto function = function_t::all().get(name);
    UTEST_REQUIRE(function);
    UTEST_REQUIRE_NOTHROW(function->config(args...));
    return function;
}

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

struct function_config_t
{
    int      m_trials{100};
    scalar_t m_central_difference_epsilon{1e-8};
    scalar_t m_convex_inequality_epsilon{1e-10};
    scalar_t m_is_convex_epsilon{1e-13};
};

[[maybe_unused]] inline auto check_function(const function_t& function, const function_config_t& config)
{
    UTEST_NAMED_CASE(function.name());

    const auto rfunction = function.clone();
    UTEST_REQUIRE(rfunction != nullptr);
    UTEST_CHECK_EQUAL(rfunction->size(), function.size());

    for (auto trial = 0; trial < config.m_trials; ++trial)
    {
        const auto x = make_random_x0(*rfunction);
        const auto z = make_random_x0(*rfunction);

        // check name
        const auto name           = rfunction->name(false);
        const auto name_with_dims = rfunction->name(true);
        UTEST_CHECK_EQUAL(name + scat("[", rfunction->size(), "D]"), name_with_dims);

        // check (sub-)gradient approximation with centering difference
        UTEST_CHECK_LESS(grad_accuracy(*rfunction, x, config.m_central_difference_epsilon),
                         config.m_central_difference_epsilon);

        // check convexity
        if (rfunction->convex())
        {
            UTEST_CHECK(is_convex(*rfunction, x, z, 20, config.m_is_convex_epsilon));
        }

        // check (sub-)gradient inequality for convex inequality
        UTEST_CHECK_GREATER_EQUAL(rfunction->strong_convexity(), 0.0);
        if (rfunction->convex())
        {
            auto       gx      = vector_t{x.size()};
            const auto fz      = (*rfunction)(z);
            const auto fx      = (*rfunction)(x, gx);
            const auto epsilon = config.m_convex_inequality_epsilon * 0.5 * (std::fabs(fx) + std::fabs(fz));
            UTEST_CHECK_GREATER_EQUAL(fz - fx + epsilon, gx.dot(z - x));
        }

        // check Hessian approximation with centering difference
        if (rfunction->smooth())
        {
            UTEST_CHECK_LESS(hess_accuracy(*rfunction, x, config.m_central_difference_epsilon),
                             config.m_central_difference_epsilon);
        }
    }
}
