#pragma once

#include <nano/stats.h>
#include <nano/measure.h>

namespace nano
{
    ///
    /// \brief accumulate time measurements for a given operation of given complexity (aka flops).
    ///
    class probe_t
    {
    public:

        probe_t(std::string basename = std::string(),
                std::string fullname = std::string(),
                const int64_t flops = 1) :
                m_basename(std::move(basename)),
                m_fullname(std::move(fullname)),
                m_flops(flops)
        {
        }

        template <typename toperator>
        void measure(const toperator& op, const int64_t count = 1)
        {
            assert(count > 0);
            const timer_t timer;
            op();
            m_timings(timer.nanoseconds().count() / count);
        }

        operator bool() const { return m_timings; }
        const auto& timings() const { return m_timings; }

        const auto& basename() const { return m_basename; }
        const auto& fullname() const { return m_fullname; }

        auto flops() const { return m_flops; }
        auto kflops() const { return m_flops / 1024; }
        auto gflops() const { return nano::gflops(flops(), nanoseconds_t(static_cast<int64_t>(timings().min()))); }

    private:

        // attributes
        std::string     m_basename;             ///<
        std::string     m_fullname;             ///<
        int64_t         m_flops;                ///< number of floating point operations per call
        stats_t         m_timings;              ///< time measurements
    };

    using probes_t = std::vector<probe_t>;
}
