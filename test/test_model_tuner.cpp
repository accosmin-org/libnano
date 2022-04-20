#include <utest/utest.h>
#include <nano/model/tuner.h>

using namespace nano;

static auto make_tuner(param_spaces_t spaces, tuner_t::callback_t callback)
{
    return tuner_t{std::move(spaces), std::move(callback)};
}

static void check_tuner(
    const tuner_t& tuner, const tensor2d_t& initial_params,
    scalar_t expected_opt_value, const tensor1d_t& expected_opt_param,
    scalar_t expected_surrogate_fit_value, const tensor1d_t& expected_surrogate_fit_param,
    scalar_t expected_surrogate_opt_value, const tensor1d_t& expected_surrogate_opt_param)
{
    const auto steps = tuner.optimize(initial_params);
    UTEST_REQUIRE_GREATER_EQUAL(steps.size(), static_cast<size_t>(initial_params.size<0>()));
    UTEST_CHECK_CLOSE(steps.rbegin()->m_value, expected_opt_value, 1e-12);
    UTEST_CHECK_CLOSE(steps.rbegin()->m_param, expected_opt_param, 1e-12);
    UTEST_CHECK_CLOSE(steps.rbegin()->m_opt_value, expected_opt_value, 1e-12);
    UTEST_CHECK_CLOSE(steps.rbegin()->m_opt_param, expected_opt_param, 1e-12);
    UTEST_CHECK_CLOSE(steps.rbegin()->m_surrogate_fit.f, expected_surrogate_fit_value, 1e-6);
    UTEST_CHECK_CLOSE(steps.rbegin()->m_surrogate_fit.x, expected_surrogate_fit_param.vector(), 1e-6);
    UTEST_CHECK_CLOSE(steps.rbegin()->m_surrogate_opt.f, expected_surrogate_opt_value, 1e-6);
    UTEST_CHECK_CLOSE(steps.rbegin()->m_surrogate_opt.x, expected_surrogate_opt_param.vector(), 1e-6);

    for (const auto& step : steps)
    {
        if (step.m_surrogate_fit.function != nullptr)
        {
            UTEST_CHECK_EQUAL(step.m_surrogate_fit.m_status, solver_state_t::status::converged);
            UTEST_CHECK_EQUAL(step.m_surrogate_opt.m_status, solver_state_t::status::converged);
        }
    }
}

UTEST_BEGIN_MODULE(test_model_tuner)

UTEST_CASE(tuner_invalid)
{
    const auto param_spaces = param_spaces_t{};
    const auto callback = [] (const tensor1d_t&) { return 0.0; };

    UTEST_CHECK_THROW(make_tuner(param_spaces, callback), std::runtime_error);
}

UTEST_CASE(tuner_optfail)
{
    const auto tuner = tuner_t
    {
        param_spaces_t
        {
            param_space_t
            {
                param_space_t::type::linear,
                make_tensor<scalar_t>(make_dims(6), 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
            }
        },
        [] (const tensor1d_t&)
        {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }
    };

    UTEST_CHECK_THROW(tuner.optimize(make_tensor<scalar_t>(make_dims(3, 1), 0.2, 0.1, 0.3)), std::runtime_error);
}

UTEST_CASE(tuner_optimize1d)
{
    const auto callback = [] (const tensor1d_t& params)
    {
        const auto x = params(0);
        return 0.78 - 1.2 * x + x * x;
    };

    auto tuner = tuner_t
    {
        param_spaces_t
        {
            param_space_t
            {
                param_space_t::type::linear,
                make_tensor<scalar_t>(make_dims(10), 0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
            }
        },
        callback
    };

    check_tuner(
        tuner, make_tensor<scalar_t>(make_dims(5, 1), 0.0, 0.4, 0.2, 0.3, 0.5),
        0.42, make_tensor<scalar_t>(make_dims(1), 0.6),
        0.00, make_tensor<scalar_t>(make_dims(3), 0.78, -1.2, 1.0),
        0.42, make_tensor<scalar_t>(make_dims(1), 0.6));

    check_tuner(
        tuner, make_tensor<scalar_t>(make_dims(2, 1), 0.0, 0.4),
        0.46, make_tensor<scalar_t>(make_dims(1), 0.4),
        0.00, tensor1d_t{},
        0.00, tensor1d_t{});

    tuner.parameter("tuner::max_iterations") = 0;
    check_tuner(
        tuner, make_tensor<scalar_t>(make_dims(5, 1), 0.0, 0.4, 0.2, 0.3, 0.5),
        0.43, make_tensor<scalar_t>(make_dims(1), 0.5),
        0.00, tensor1d_t{},
        0.00, tensor1d_t{});
}

UTEST_CASE(tuner_optimize2d)
{
    const auto callback = [] (const tensor1d_t& params)
    {
        const auto x = params(0), y = std::log10(params(1));
        return (x - 0.1 * y) * (x - 0.1 * y) + (y - 1.0) * (y - 1.0) + 0.7;
    };

    auto tuner = tuner_t
    {
        param_spaces_t
        {
            param_space_t
            {
                param_space_t::type::linear, make_tensor<scalar_t>(make_dims(7), 0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0)
            },
            param_space_t
            {
                param_space_t::type::log10, make_tensor<scalar_t>(make_dims(6), 1e-6, 1e-3, 1e+0, 3e+0, 1e+1, 1e+2)
            }
        },
        callback
    };

    check_tuner(
        tuner, make_tensor<scalar_t>(make_dims(12, 2),
            0.0, 1e-3, 0.5, 3e-1, 0.4, 2e+1, 0.5, 1e+1, 1.0, 7e-1, 0.3, 1e-2,
            0.9, 1e-1, 0.7, 1e-4, 0.5, 1e+0, 0.9, 1e+1, 0.8, 2e+1, 0.6, 1e+0),
        0.7, make_tensor<scalar_t>(make_dims(2), 0.1, 10.0),
        0.0, make_tensor<scalar_t>(make_dims(6), 1.7, 0.0, -2.0, 1.0, -0.2, 1.01),
        0.7, make_tensor<scalar_t>(make_dims(2), 0.1, 1.0));

    check_tuner(
        tuner, make_tensor<scalar_t>(make_dims(6, 2),
            0.0, 1e-3, 0.5, 1e-1, 0.8, 1e+1, 0.5, 1e+0, 0.9, 1e+2, 0.4, 1e+1),
        0.7, make_tensor<scalar_t>(make_dims(2), 0.1, 10.0),
        0.0, make_tensor<scalar_t>(make_dims(6), 1.7, 0.0, -2.0, 1.0, -0.2, 1.01),
        0.7, make_tensor<scalar_t>(make_dims(2), 0.1, 1.0));

    tuner.parameter("tuner::max_iterations") = 0;
    check_tuner(
        tuner, make_tensor<scalar_t>(make_dims(6, 2),
            0.0, 1e-3, 0.5, 1e-1, 0.8, 1e+1, 0.5, 1e+0, 0.9, 1e+2, 0.4, 1e+1),
        0.79, make_tensor<scalar_t>(make_dims(2), 0.4, 10.0),
        0.00, tensor1d_t{},
        0.00, tensor1d_t{});
}

UTEST_END_MODULE()
