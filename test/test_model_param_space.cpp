#include <nano/model/param_space.h>
#include <utest/utest.h>

using namespace nano;

static auto make_param_space(param_space_t::type type, tensor1d_t param_grid)
{
    return param_space_t{type, std::move(param_grid)};
}

UTEST_BEGIN_MODULE(test_model_param_space)

UTEST_CASE(param_space_invalid)
{
    const auto param_gridNON = tensor1d_t{};
    const auto param_gridENG = make_tensor<scalar_t>(make_dims(1), 1.0);
    const auto param_gridNEG = make_tensor<scalar_t>(make_dims(2), -1.0, +1.0);
    const auto param_gridDUP = make_tensor<scalar_t>(make_dims(3), -1.0, +1.0, +1.0);
    const auto param_gridORD = make_tensor<scalar_t>(make_dims(4), -1.0, +2.0, +1.0, +3.0);

    UTEST_CHECK_THROW(make_param_space(param_space_t::type::log10, param_gridNEG), std::runtime_error);

    UTEST_CHECK_THROW(make_param_space(param_space_t::type::log10, param_gridNON), std::runtime_error);
    UTEST_CHECK_THROW(make_param_space(param_space_t::type::linear, param_gridNON), std::runtime_error);

    UTEST_CHECK_THROW(make_param_space(param_space_t::type::log10, param_gridENG), std::runtime_error);
    UTEST_CHECK_THROW(make_param_space(param_space_t::type::linear, param_gridENG), std::runtime_error);

    UTEST_CHECK_THROW(make_param_space(param_space_t::type::log10, param_gridDUP), std::runtime_error);
    UTEST_CHECK_THROW(make_param_space(param_space_t::type::linear, param_gridDUP), std::runtime_error);

    UTEST_CHECK_THROW(make_param_space(param_space_t::type::log10, param_gridORD), std::runtime_error);
    UTEST_CHECK_THROW(make_param_space(param_space_t::type::linear, param_gridORD), std::runtime_error);
}

UTEST_CASE(param_space_log10)
{
    const auto type  = param_space_t::type::log10;
    const auto space = param_space_t{type, make_tensor<scalar_t>(make_dims(4), 1e-6, 1e-3, 1e+1, 1e+2)};

    UTEST_CHECK_CLOSE(space.to_surrogate(1e-5), -5.0, 1e-12);
    UTEST_CHECK_CLOSE(space.to_surrogate(1e+0), +0.0, 1e-12);
    UTEST_CHECK_CLOSE(space.to_surrogate(1e+2), +2.0, 1e-12);
    UTEST_CHECK_THROW(space.to_surrogate(3e-7), std::runtime_error);
    UTEST_CHECK_THROW(space.to_surrogate(1e+7), std::runtime_error);

    UTEST_CHECK_CLOSE(space.from_surrogate(-7.0), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(-6.0), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(-1.0), 1e-1, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+1.0), 1e+1, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+2.0), 1e+2, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+3.0), 1e+2, 1e-12);

    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(-7.0), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(-6.0), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(-3.1), 1e-3, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+0.5), 1e+1, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+1.6), 1e+2, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+2.1), 1e+2, 1e-12);
}

UTEST_CASE(param_space_linear)
{
    const auto type  = param_space_t::type::linear;
    const auto space = param_space_t{type, make_tensor<scalar_t>(make_dims(4), 0.1, 0.2, 0.5, 1.0)};

    UTEST_CHECK_CLOSE(space.to_surrogate(0.10), +0.0, 1e-12);
    UTEST_CHECK_CLOSE(space.to_surrogate(0.55), +0.5, 1e-12);
    UTEST_CHECK_CLOSE(space.to_surrogate(1.00), +1.0, 1e-12);
    UTEST_CHECK_THROW(space.to_surrogate(0.01), std::runtime_error);
    UTEST_CHECK_THROW(space.to_surrogate(1.01), std::runtime_error);

    UTEST_CHECK_CLOSE(space.from_surrogate(-1.0), 0.10, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+0.0), 0.10, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+0.5), 0.55, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+1.0), 1.00, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+2.0), 1.00, 1e-12);

    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(-1.0), 0.10, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+0.0), 0.10, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+0.1), 0.20, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+0.5), 0.50, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+1.0), 1.00, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+1.1), 1.00, 1e-12);
}

UTEST_END_MODULE()
