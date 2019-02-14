#pragma once

#include <nano/timer.h>
#include <nano/numeric.h>

namespace nano
{
    ///
    /// \brief robustly measure a function call (in the given time units).
    ///
    template <typename tunits, typename toperator>
    tunits measure(const toperator& op, const std::size_t trials,
        const std::size_t min_trial_iterations = 1,
        const microseconds_t min_trial_duration = microseconds_t(1000))
    {
        const auto run_opx = [&] (const size_t times)
        {
            const timer_t timer;
            for (std::size_t i = 0; i < times; ++ i)
            {
                    op();
            }
            return timer.microseconds();
        };

        const auto run_trial = [&] (const size_t times)
        {
            return picoseconds_t(nano::idiv(run_opx(times).count() * 1000 * 1000, times));
        };

        // calibrate the number of function calls to achieve the minimum time resolution
        std::size_t trial_iterations = std::max(std::size_t(1), min_trial_iterations);
        for (microseconds_t usecs(0); usecs < min_trial_duration; trial_iterations *= 2)
        {
            usecs = run_opx(trial_iterations);
        }

        // measure multiple times for robustness
        picoseconds_t duration = run_trial(trial_iterations);
        for (std::size_t t = 1; t < trials; ++ t)
        {
            duration = std::min(duration, run_trial(trial_iterations));
        }

        return std::chrono::duration_cast<tunits>(duration);
    }

    ///
    /// \brief compute GFLOPS (giga floating point operations per seconds)
    ///     given the number of FLOPs run in the given duration.
    ///
    template <typename tinteger, typename tduration>
    int64_t gflops(const tinteger flops, const tduration& duration)
    {
        const auto div = static_cast<int64_t>(std::chrono::duration_cast<picoseconds_t>(duration).count());
        return nano::idiv(static_cast<int64_t>(flops) * 1000, std::max(div, int64_t(1)));
    }
}
