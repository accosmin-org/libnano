#include <nano/tune.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_tune)

UTEST_CASE(tune1d_geom)
{
    const auto evaluator = [&] (const scalar_t x)
    {
        return (x - 1.0) * (x - 1.0);
    };

    {
        const auto optimum = nano::geom_tune(evaluator, 0.5, 0.7, 1.01);
        UTEST_CHECK_CLOSE(std::get<0>(optimum), 0.0, 1e-3);
        UTEST_CHECK_CLOSE(std::get<1>(optimum), 1.0, 1e-2);
        UTEST_CHECK_CLOSE(std::get<0>(optimum), evaluator(std::get<1>(optimum)), 1e-8);
    }
    {
        const auto optimum = nano::geom_tune(evaluator, 2.3, 3.7, 1.01);
        UTEST_CHECK_CLOSE(std::get<0>(optimum), 0.0, 1e-3);
        UTEST_CHECK_CLOSE(std::get<1>(optimum), 1.0, 1e-2);
        UTEST_CHECK_CLOSE(std::get<0>(optimum), evaluator(std::get<1>(optimum)), 1e-8);
    }
}

UTEST_CASE(tune1d_pow10)
{
    const auto space = pow10_space_t{-1.7, +1.1};
    const auto evaluator = [&] (const scalar_t x)
    {
        UTEST_CHECK_LESS_EQUAL(std::pow(10.0, space.min()), x);
        UTEST_CHECK_LESS_EQUAL(x, std::pow(10.0, space.max()));
        return (x - 1.0) * (x - 1.0);
    };

    const auto optimum = nano::grid_tune(space, evaluator, 7, 5);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), 0.0, 1e-3);
    UTEST_CHECK_CLOSE(std::get<1>(optimum), 1.0, 1e-2);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), evaluator(std::get<1>(optimum)), 1e-8);
}

UTEST_CASE(tune1d_linear)
{
    const auto space = linear_space_t{-5.7, +9.1};
    const auto evaluator = [&] (const scalar_t x)
    {
        UTEST_CHECK_LESS_EQUAL(space.min(), x);
        UTEST_CHECK_LESS_EQUAL(x, space.max());
        return (x < 0.0) ? std::numeric_limits<scalar_t>::quiet_NaN() : ((x - 1.0) * (x - 1.0) + 1.3);
    };

    const auto optimum = nano::grid_tune(space, evaluator, 7, 7);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), 1.3, 1e-3);
    UTEST_CHECK_CLOSE(std::get<1>(optimum), 1.0, 1e-2);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), evaluator(std::get<1>(optimum)), 1e-8);
}

UTEST_CASE(tune1d_domain_error)
{
    const auto space = linear_space_t{-2.0, -1.0};
    const auto evaluator = [&] (const scalar_t x)
    {
        return (x < 0.0) ? std::numeric_limits<scalar_t>::quiet_NaN() : std::log(1 + x * x);
    };

    UTEST_CHECK_THROW(nano::grid_tune(space, evaluator, 7, 7), std::runtime_error);
}

UTEST_CASE(tune2d_mixing)
{
    const auto space1 = pow10_space_t{-2.1, +2.3};
    const auto space2 = linear_space_t{-5.7, +9.1};
    const auto evaluator = [&] (const scalar_t x, const scalar_t y)
    {
        UTEST_CHECK_LESS_EQUAL(std::pow(10.0, space1.min()), x);
        UTEST_CHECK_LESS_EQUAL(x, std::pow(10.0, space1.max()));

        UTEST_CHECK_LESS_EQUAL(space2.min(), y);
        UTEST_CHECK_LESS_EQUAL(y, space2.max());

        return (x - 1.0) * (x - 1.0) + std::log(1 + (x - y + 0.5) * (x - y + 0.5)) + 1.3;
    };

    const auto optimum = nano::grid_tune(space1, space2, evaluator, 7, 7);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), 1.3, 1e-3);
    UTEST_CHECK_CLOSE(std::get<1>(optimum), 1.0, 1e-2);
    UTEST_CHECK_CLOSE(std::get<2>(optimum), 1.5, 1e-2);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), evaluator(std::get<1>(optimum), std::get<2>(optimum)), 1e-8);
}

UTEST_CASE(tune2d_domain_error)
{
    const auto space1 = pow10_space_t{-2.1, +2.3};
    const auto space2 = linear_space_t{-5.7, +9.1};
    const auto evaluator = [&] (const scalar_t, const scalar_t)
    {
        return std::numeric_limits<scalar_t>::quiet_NaN();
    };

    UTEST_CHECK_THROW(nano::grid_tune(space1, space2, evaluator, 7, 7), std::runtime_error);
}

UTEST_END_MODULE()
