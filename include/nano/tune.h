#pragma once

#include <cmath>
#include <tuple>
#include <vector>
#include <cassert>
#include <algorithm>
#include <nano/scalar.h>

namespace nano
{
    ///
    /// \brief the search interval used for tuning hyper-parameters.
    ///
    class tuning_space_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        tuning_space_t() = default;

        ///
        /// \brief constructor
        ///
        tuning_space_t(const scalar_t min, const scalar_t max) :
            m_min(min),
            m_max(max)
        {
            assert(m_min < m_max);
        }

        ///
        /// \brief adjust the search interval around the given optimum
        ///
        void refine(const scalar_t optimum)
        {
            assert(m_min <= optimum && optimum <= m_max);
            const auto var = (m_max - m_min) / 4;
            m_min = std::max(m_min, optimum - var);
            m_max = std::min(m_max, optimum + var);
        }

        ///
        /// \brief return the current search interval
        ///
        auto min() const { return m_min; }
        auto max() const { return m_max; }

        ///
        /// \brief clamp the given (trial) value to the current search interval
        ///
        auto clamp(const scalar_t value) const
        {
            return std::max(min(), std::min(value, max()));
        }

    private:

        // attributes
        scalar_t    m_min{-6};  ///< minimum of the current search interval
        scalar_t    m_max{+6};  ///< maximum of the current search interval
    };

    ///
    /// \brief the search interval using power of 10s for tuning hyper-parameters.
    ///
    class pow10_space_t : public tuning_space_t
    {
    public:

        using tuning_space_t::tuning_space_t;

        ///
        /// \brief adjust the search interval around the given optimum
        ///
        void refine(const scalar_t pow10_optimum)
        {
            assert(pow10_optimum > 0);
            tuning_space_t::refine(std::log10(pow10_optimum));
        }

        ///
        /// \brief generate a list of hyper-parameter values to evaluate
        ///
        auto generate(const int count) const
        {
            assert(count > 3);

            std::vector<scalar_t> trials(static_cast<size_t>(count));
            for (size_t i = 0; i < trials.size(); ++ i)
            {
                const auto trial = min() + static_cast<scalar_t>(i) * (max() - min()) / (count - 1);
                trials[i] = std::pow(scalar_t(10), clamp(trial));
            }

            return trials;
        }
    };

    ///
    /// \brief the search interval using linear mapping for tuning hyper-parameters.
    ///
    class linear_space_t : public tuning_space_t
    {
    public:

        using tuning_space_t::tuning_space_t;

        ///
        /// \brief adjust the search interval around the given optimum
        ///
        void refine(const scalar_t optimum)
        {
            tuning_space_t::refine(optimum);
        }

        ///
        /// \brief generate a list of hyper-parameter values to evaluate
        ///
        auto generate(const int count) const
        {
            assert(count > 3);

            std::vector<scalar_t> trials(static_cast<size_t>(count));
            for (size_t i = 0; i < trials.size(); ++ i)
            {
                const auto trial = min() + static_cast<scalar_t>(i) * (max() - min()) / (count - 1);
                trials[i] = clamp(trial);
            }

            return trials;
        }
    };

    ///
    /// \brief coarse-to-fine tuning of a continuous hyper-parameter.
    ///     the tuning is performed in steps by sampling with finer and finer step size around the current optimum.
    ///
    template <typename tspace1, typename tevaluator>
    std::tuple<scalar_t, scalar_t> tune(tspace1 space1, const tevaluator& evaluator,
        const int maximum_trials_per_step, const int steps)
    {
        assert(steps > 0 && maximum_trials_per_step > 3);

        std::vector<std::tuple<scalar_t, scalar_t>> results;
        for (int step = 0; step < steps; ++ step)
        {
            for (const auto param1 : space1.generate(maximum_trials_per_step))
            {
                const auto value = static_cast<scalar_t>(evaluator(param1));
                results.emplace_back(value, param1);
            }

            if (step + 1 < steps)
            {
                assert(!results.empty());
                const auto it_min = std::min_element(results.begin(), results.end());
                space1.refine(std::get<1>(*it_min));
            }
        }

        assert(!results.empty());
        return *std::min_element(results.begin(), results.end());
    }

    ///
    /// \brief coarse-to-fine tuning of two continuous hyper-parameters.
    ///     the tuning is performed in steps by sampling with finer and finer step size around the current optimum.
    ///
    template <typename tspace1, typename tspace2, typename tevaluator>
    std::tuple<scalar_t, scalar_t, scalar_t> tune(tspace1 space1, tspace2 space2, const tevaluator& evaluator,
        const int maximum_trials_per_step, const int steps)
    {
        assert(steps > 0 && maximum_trials_per_step > 3);

        std::vector<std::tuple<scalar_t, scalar_t, scalar_t>> results;
        for (int step = 0; step < steps; ++ step)
        {
            for (const auto param1 : space1.generate(maximum_trials_per_step))
            {
                for (const auto param2 : space2.generate(maximum_trials_per_step))
                {
                    const auto value = static_cast<scalar_t>(evaluator(param1, param2));
                    results.emplace_back(value, param1, param2);
                }
            }

            if (step + 1 < steps)
            {
                assert(!results.empty());
                const auto it_min = std::min_element(results.begin(), results.end());
                space1.refine(std::get<1>(*it_min));
                space2.refine(std::get<2>(*it_min));
            }
        }

        assert(!results.empty());
        return *std::min_element(results.begin(), results.end());
    }
}
