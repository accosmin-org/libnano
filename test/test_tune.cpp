#include <nano/tune.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_tune)

UTEST_CASE(equal)
{
    using ::nano::detail::equal;
    using tuple1_t = std::tuple<scalar_t>;
    using tuple2_t = std::tuple<scalar_t, scalar_t>;

    const auto epsilon = std::sqrt(std::numeric_limits<scalar_t>::epsilon());
    const auto tiny = 0.25 * epsilon;

    UTEST_CHECK_EQUAL(equal(tuple1_t{1.0}, tuple1_t{1.1}, epsilon), false);
    UTEST_CHECK_EQUAL(equal(tuple1_t{1.0}, tuple1_t{1.0 + tiny}, epsilon), true);

    UTEST_CHECK_EQUAL(equal(tuple2_t{1.0, 2.0}, tuple2_t{1.0, 2.1}, epsilon), false);
    UTEST_CHECK_EQUAL(equal(tuple2_t{1.1, 2.0}, tuple2_t{1.0, 2.0}, epsilon), false);
    UTEST_CHECK_EQUAL(equal(tuple2_t{1.0, 2.0}, tuple2_t{1.0, 2.0}, epsilon), true);
    UTEST_CHECK_EQUAL(equal(tuple2_t{1.0, 2.0}, tuple2_t{1.0, 2.0 - tiny}, epsilon), true);
    UTEST_CHECK_EQUAL(equal(tuple2_t{1.0 + tiny, 2.0}, tuple2_t{1.0 - tiny, 2.0}, epsilon), true);
    UTEST_CHECK_EQUAL(equal(tuple2_t{1.0 + tiny, 2.0}, tuple2_t{1.0, 2.0 - tiny}, epsilon), true);
}

UTEST_CASE(checked)
{
    using ::nano::detail::checked;
    using tuple1_t = std::tuple<scalar_t>;

    const auto epsilon = std::sqrt(std::numeric_limits<scalar_t>::epsilon());
    const auto tiny = 0.25 * epsilon;

    std::vector<tuple1_t> history;
    UTEST_CHECK_EQUAL(history.size(), 0U);
    UTEST_CHECK_EQUAL(checked(history, tuple1_t{1.0}, epsilon), false);
    UTEST_CHECK_EQUAL(history.size(), 1U);
    UTEST_CHECK_EQUAL(checked(history, tuple1_t{1.1}, epsilon), false);
    UTEST_CHECK_EQUAL(history.size(), 2U);
    UTEST_CHECK_EQUAL(checked(history, tuple1_t{1.1 - tiny}, epsilon), true);
    UTEST_CHECK_EQUAL(history.size(), 2U);
    UTEST_CHECK_EQUAL(checked(history, tuple1_t{1.0 + tiny}, epsilon), true);
    UTEST_CHECK_EQUAL(history.size(), 2U);
    UTEST_CHECK_EQUAL(checked(history, tuple1_t{1.2 - tiny}, epsilon), false);
    UTEST_CHECK_EQUAL(history.size(), 3U);
    UTEST_CHECK_EQUAL(checked(history, tuple1_t{1.3}, epsilon), false);
    UTEST_CHECK_EQUAL(history.size(), 4U);
}

UTEST_CASE(space_pow10)
{
    auto space = pow10_space_t{-2.0, +2.0};
    UTEST_CHECK_CLOSE(space.min(), -2.0, 1e-12);
    UTEST_CHECK_CLOSE(space.max(), +2.0, 1e-12);
    UTEST_CHECK_CLOSE(space.clamp(-2.1), -2.0, 1e-12);
    UTEST_CHECK_CLOSE(space.clamp(-0.1), -0.1, 1e-12);
    UTEST_CHECK_CLOSE(space.clamp(+2.3), +2.0, 1e-12);

    const auto trials = space.generate(9);
    UTEST_REQUIRE_EQUAL(trials.size(), 9U);
    UTEST_CHECK_CLOSE(trials[0U], std::pow(10.0, -2.0), 1e-12);
    UTEST_CHECK_CLOSE(trials[1U], std::pow(10.0, -1.5), 1e-12);
    UTEST_CHECK_CLOSE(trials[2U], std::pow(10.0, -1.0), 1e-12);
    UTEST_CHECK_CLOSE(trials[3U], std::pow(10.0, -0.5), 1e-12);
    UTEST_CHECK_CLOSE(trials[4U], std::pow(10.0, +0.0), 1e-12);
    UTEST_CHECK_CLOSE(trials[5U], std::pow(10.0, +0.5), 1e-12);
    UTEST_CHECK_CLOSE(trials[6U], std::pow(10.0, +1.0), 1e-12);
    UTEST_CHECK_CLOSE(trials[7U], std::pow(10.0, +1.5), 1e-12);
    UTEST_CHECK_CLOSE(trials[8U], std::pow(10.0, +2.0), 1e-12);

    UTEST_CHECK_THROW(space.refine(std::numeric_limits<scalar_t>::quiet_NaN()), std::runtime_error);
    UTEST_CHECK_THROW(space.refine(-1.0), std::runtime_error);

    UTEST_CHECK_NOTHROW(space.refine(std::pow(10.0, 0.5)));
    UTEST_CHECK_CLOSE(space.min(), -0.5, 1e-12);
    UTEST_CHECK_CLOSE(space.max(), +1.5, 1e-12);
}

UTEST_CASE(space_linear)
{
    auto space = linear_space_t{-2.0, +2.0};
    UTEST_CHECK_CLOSE(space.min(), -2.0, 1e-12);
    UTEST_CHECK_CLOSE(space.max(), +2.0, 1e-12);
    UTEST_CHECK_CLOSE(space.clamp(-2.1), -2.0, 1e-12);
    UTEST_CHECK_CLOSE(space.clamp(-0.1), -0.1, 1e-12);
    UTEST_CHECK_CLOSE(space.clamp(+2.3), +2.0, 1e-12);

    const auto trials = space.generate(5);
    UTEST_REQUIRE_EQUAL(trials.size(), 5U);
    UTEST_CHECK_CLOSE(trials[0U], -2.0, 1e-12);
    UTEST_CHECK_CLOSE(trials[1U], -1.0, 1e-12);
    UTEST_CHECK_CLOSE(trials[2U], +0.0, 1e-12);
    UTEST_CHECK_CLOSE(trials[3U], +1.0, 1e-12);
    UTEST_CHECK_CLOSE(trials[4U], +2.0, 1e-12);

    UTEST_CHECK_THROW(space.refine(std::numeric_limits<scalar_t>::quiet_NaN()), std::runtime_error);
    UTEST_CHECK_THROW(space.refine(std::numeric_limits<scalar_t>::infinity()), std::runtime_error);

    UTEST_CHECK_NOTHROW(space.refine(+1.7));
    UTEST_CHECK_CLOSE(space.min(), +0.7, 1e-12);
    UTEST_CHECK_CLOSE(space.max(), +2.0, 1e-12);
}

UTEST_CASE(tune1d_pow10)
{
    const auto space = pow10_space_t{-1.7, +1.1};
    const auto evaluator = [&] (const scalar_t x)
    {
        UTEST_CHECK_LESS_EQUAL(std::pow(10.0, space.min()), x);
        UTEST_CHECK_LESS_EQUAL(x, std::pow(10.0, space.max()));
        const auto value = (x - 1.0) * (x - 1.0);
        return std::make_tuple(value, 2.0 * x);
    };

    const auto optimum = nano::grid_tune(space, evaluator, 7, 7);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), 1.0, 1e-3);
    UTEST_CHECK_CLOSE(std::get<1>(optimum), 0.0, 1e-2);
    UTEST_CHECK_CLOSE(std::get<2>(optimum), 2.0, 1e-2);
}

UTEST_CASE(tune1d_linear)
{
    const auto space = linear_space_t{-5.7, +9.1};
    const auto evaluator = [&] (const scalar_t x)
    {
        UTEST_CHECK_LESS_EQUAL(space.min(), x);
        UTEST_CHECK_LESS_EQUAL(x, space.max());
        const auto value = (x < 0.0) ? std::numeric_limits<scalar_t>::quiet_NaN() : ((x - 1.0) * (x - 1.0) + 1.3);
        return std::make_tuple(value);
    };

    const auto optimum = nano::grid_tune(space, evaluator, 7, 10);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), 1.0, 1e-3);
    UTEST_CHECK_CLOSE(std::get<1>(optimum), 1.3, 1e-2);
}

UTEST_CASE(tune1d_domain_error)
{
    const auto space = linear_space_t{-2.0, -1.0};
    const auto evaluator = [&] (const scalar_t x)
    {
        const auto value = (x < 0.0) ? std::numeric_limits<scalar_t>::quiet_NaN() : std::log1p(x * x);
        return std::make_tuple(value);
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

        const auto value = (x - 1.0) * (x - 1.0) + std::log1p((x - y + 0.5) * (x - y + 0.5)) + 1.3;
        return std::make_tuple(value, x + y, x - y);
    };

    const auto optimum = nano::grid_tune(space1, space2, evaluator, 7, 7);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), 1.0, 1e-2);
    UTEST_CHECK_CLOSE(std::get<1>(optimum), 1.5, 1e-2);
    UTEST_CHECK_CLOSE(std::get<2>(optimum), 1.3, 1e-3);
    UTEST_CHECK_CLOSE(std::get<3>(optimum), +2.5, 1e-2);
    UTEST_CHECK_CLOSE(std::get<4>(optimum), -0.5, 1e-2);
}

UTEST_CASE(tune2d_domain_error)
{
    const auto space1 = pow10_space_t{-2.1, +2.3};
    const auto space2 = linear_space_t{-5.7, +9.1};
    const auto evaluator = [&] (const scalar_t, const scalar_t)
    {
        return std::make_tuple(std::numeric_limits<scalar_t>::quiet_NaN());
    };

    UTEST_CHECK_THROW(nano::grid_tune(space1, space2, evaluator, 7, 7), std::runtime_error);
}

UTEST_CASE(tune3d_mixing)
{
    const auto space1 = pow10_space_t{-2.1, +2.3};
    const auto space2 = linear_space_t{-5.7, +9.1};
    const auto space3 = linear_space_t{-4.7, -1.1};
    const auto evaluator = [&] (const scalar_t x, const scalar_t y, const scalar_t z)
    {
        UTEST_CHECK_LESS_EQUAL(std::pow(10.0, space1.min()), x);
        UTEST_CHECK_LESS_EQUAL(x, std::pow(10.0, space1.max()));

        UTEST_CHECK_LESS_EQUAL(space2.min(), y);
        UTEST_CHECK_LESS_EQUAL(y, space2.max());

        UTEST_CHECK_LESS_EQUAL(space3.min(), z);
        UTEST_CHECK_LESS_EQUAL(z, space3.max());

        const auto value = (x - 1.0) * (x - 1.0) + std::log1p((x - y + 0.5) * (x - y + 0.5)) + (z + 2.0) * (z + 2.0) + 0.5;
        return std::make_tuple(value);
    };

    const auto optimum = nano::grid_tune(space1, space2, space3, evaluator, 7, 7);
    UTEST_CHECK_CLOSE(std::get<0>(optimum), 1.0, 1e-2);
    UTEST_CHECK_CLOSE(std::get<1>(optimum), 1.5, 1e-2);
    UTEST_CHECK_CLOSE(std::get<2>(optimum), -2.0, 1e-2);
    UTEST_CHECK_CLOSE(std::get<3>(optimum), 0.5, 1e-3);
}

UTEST_CASE(tune3d_domain_error)
{
    const auto space1 = pow10_space_t{-2.1, +2.3};
    const auto space2 = linear_space_t{-5.7, +9.1};
    const auto space3 = linear_space_t{-3.7, +8.1};
    const auto evaluator = [&] (const scalar_t, const scalar_t, const scalar_t)
    {
        return std::make_tuple(std::numeric_limits<scalar_t>::quiet_NaN());
    };

    UTEST_CHECK_THROW(nano::grid_tune(space1, space2, space3, evaluator, 7, 7), std::runtime_error);
}

UTEST_END_MODULE()
