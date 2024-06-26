#include <nano/core/chrono.h>
#include <utest/utest.h>

UTEST_BEGIN_MODULE(test_chrono)

UTEST_CASE(elapsed)
{
    UTEST_CHECK_EQUAL(nano::elapsed(0), "000ms");
    UTEST_CHECK_EQUAL(nano::elapsed(999), "999ms");
    UTEST_CHECK_EQUAL(nano::elapsed(1001), "01s:001ms");
    UTEST_CHECK_EQUAL(nano::elapsed(59999), "59s:999ms");
    UTEST_CHECK_EQUAL(nano::elapsed(60000), "01m:00s:000ms");
    UTEST_CHECK_EQUAL(nano::elapsed(600001), "10m:00s:001ms");
    UTEST_CHECK_EQUAL(nano::elapsed(3600000), "01h:00m:00s:000ms");
    UTEST_CHECK_EQUAL(nano::elapsed(3600000 * 24), "1d:00h:00m:00s:000ms");
    UTEST_CHECK_EQUAL(nano::elapsed(3600000 * 48 + 17 * 3600000 + 52 * 60000 + 17600), "2d:17h:52m:17s:600ms");
}

UTEST_CASE(timer)
{
    auto timer = nano::timer_t{};

    timer.reset();
    UTEST_CHECK_LESS(timer.nanoseconds().count(), 10000000);
    UTEST_CHECK_LESS(timer.microseconds().count(), 1000000);
    UTEST_CHECK_LESS(timer.milliseconds().count(), 1000);
    UTEST_CHECK_EQUAL(timer.seconds().count(), 0);
    UTEST_CHECK_EQUAL(timer.elapsed().empty(), false);
}

UTEST_CASE(measure)
{
    const auto op = []()
    {
        const auto volatile value = std::sqrt(std::fabs(std::sin(2.0)) + std::cos(3.0) * std::cos(3.0));
        return value;
    };

    for (auto trials : {1, 2, 4})
    {
        const auto min_trial_iterations = 1;
        const auto min_trial_duration   = nano::microseconds_t{100};
        const auto duration = nano::measure<nano::milliseconds_t>(op, trials, min_trial_iterations, min_trial_duration);
        UTEST_CHECK_GREATER_EQUAL(duration.count(), 0);
    }
}

UTEST_CASE(gflops)
{
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::seconds_t(1)), 0);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::milliseconds_t(1)), 0);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::microseconds_t(1)), 0);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(100)), 0);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(10)), 4);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::nanoseconds_t(1)), 42);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(100)), 420);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(10)), 4200);
    UTEST_CHECK_EQUAL(nano::gflops(42, nano::picoseconds_t(1)), 42000);
}

UTEST_END_MODULE()
