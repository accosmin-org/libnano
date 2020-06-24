#pragma once

#include <cmath>
#include <tuple>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <nano/scalar.h>

namespace nano
{
    // TODO: either throw an exception if the optimum value is at the boundary or
    //  automatically extend the search domain!

    namespace detail
    {
        template <size_t offset, size_t index, typename tsource, typename tdestination>
        void move_tuple(tsource&& source, tdestination&& destination)
        {
            std::get<offset + index>(destination) = std::move(std::get<index>(source));
            if constexpr (index > 0U)
            {
                move_tuple<offset, index - 1U>(source, destination);
            }
        }

        template <typename toptimum, typename tparams, typename tevaluated>
        void update(toptimum& optimum, const tparams& params, tevaluated&& evaluated)
        {
            constexpr size_t psize = std::tuple_size<tparams>::value;
            constexpr size_t esize = std::tuple_size<tevaluated>::value;

            const auto value = std::get<0>(evaluated);
            const auto best_value = std::get<psize>(optimum);
            if (std::isfinite(value) && value < best_value)
            {
                move_tuple<0U, psize - 1U>(params, optimum);
                move_tuple<psize, esize - 1U>(evaluated, optimum);
            }
        }

        template <size_t index, typename tparams>
        bool equal(const tparams& params1, const tparams& params2, const scalar_t epsilon)
        {
            if (std::fabs(std::get<index>(params1) - std::get<index>(params2)) < epsilon)
            {
                if constexpr (index > 0U)
                {
                    return equal<index - 1>(params1, params2, epsilon);
                }
                else
                {
                    return true;
                }
            }
            else
            {
                return false;
            }
        }

        template <typename tparams>
        bool equal(const tparams& params1, const tparams& params2,
            const scalar_t epsilon = std::sqrt(std::numeric_limits<scalar_t>::epsilon()))
        {
            constexpr size_t psize = std::tuple_size<tparams>::value;

            return equal<psize - 1U>(params1, params2, epsilon);
        }

        template <typename tparams>
        bool checked(std::vector<tparams>& history, const tparams& params,
            const scalar_t epsilon = std::sqrt(std::numeric_limits<scalar_t>::epsilon()))
        {
            const auto it = std::find_if(history.begin(), history.end(), [&] (const tparams& old_params)
            {
                return equal(old_params, params, epsilon);
            });

            if (it == history.end())
            {
                history.push_back(params);
                return false;
            }
            else
            {
                return true;
            }
        }
    }

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
            if (!std::isfinite(optimum))
            {
                throw std::runtime_error("invalid tuning domain for the search space");
            }
            assert(m_min <= optimum && optimum <= m_max);
            const auto var = (m_max - m_min) / 4;
            m_min = std::max(m_min, optimum - var);
            m_max = std::min(m_max, optimum + var);
        }

        ///
        /// \brief return the current search interval
        ///
        [[nodiscard]] auto min() const { return m_min; }
        [[nodiscard]] auto max() const { return m_max; }

        ///
        /// \brief clamp the given (trial) value to the current search interval
        ///
        [[nodiscard]] auto clamp(const scalar_t value) const
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
            if (!std::isfinite(pow10_optimum))
            {
                throw std::runtime_error("invalid tuning domain for the search space");
            }
            tuning_space_t::refine(std::log10(pow10_optimum));
        }

        ///
        /// \brief generate a list of hyper-parameter values to evaluate
        ///
        [[nodiscard]] auto generate(const int count) const
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
        [[nodiscard]] auto generate(const int count) const
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
    /// NB: the evaluator is called with (param1) and returns a tuple with:
    ///         <value_for_param1, others_for_param1...>
    /// NB: the final returned value is a tuple with:
    ///         <optimum_param1, value_for_optimum_param1, others_for_optimum_param1...>
    ///
    template <typename tspace1, typename tevaluator>
    auto grid_tune(tspace1 space1,
        const tevaluator& evaluator, const int maximum_trials_per_step, const int steps)
    {
        assert(steps > 0 && maximum_trials_per_step > 3);

        std::vector<std::tuple<scalar_t>> tried;

        decltype(std::tuple_cat(std::make_tuple(scalar_t{}), evaluator(0))) optimum;
        std::get<0>(optimum) = std::numeric_limits<scalar_t>::quiet_NaN();
        std::get<1>(optimum) = std::numeric_limits<scalar_t>::max();
        for (int step = 0; step < steps; ++ step)
        {
            for (const auto param1 : space1.generate(maximum_trials_per_step))
            {
                const auto params = std::make_tuple(param1);
                if (!detail::checked(tried, params))
                {
                    detail::update(optimum, params, evaluator(param1));
                }
            }

            if (step + 1 < steps)
            {
                space1.refine(std::get<0>(optimum));
            }
        }

        return optimum;
    }

    ///
    /// \brief coarse-to-fine tuning of two continuous hyper-parameters.
    ///     the tuning is performed in steps by sampling with finer and finer step size around the current optimum.
    ///
    /// NB: the evaluator is called with (param1, param2) and returns a tuple with:
    ///         <value_for_param1_2, others_for_param1_2...>
    /// NB: the final returned value is a tuple with:
    ///         <optimum_param1, optimum_param2, value_for_optimum_param1_2, others_for_optimum_param1_2...>
    ///
    template <typename tspace1, typename tspace2, typename tevaluator>
    auto grid_tune(tspace1 space1, tspace2 space2,
        const tevaluator& evaluator, const int maximum_trials_per_step, const int steps)
    {
        assert(steps > 0 && maximum_trials_per_step > 3);

        std::vector<std::tuple<scalar_t, scalar_t>> tried;

        decltype(std::tuple_cat(std::make_tuple(scalar_t{}, scalar_t{}), evaluator(0, 0))) optimum;
        std::get<0>(optimum) = std::numeric_limits<scalar_t>::quiet_NaN();
        std::get<1>(optimum) = std::numeric_limits<scalar_t>::quiet_NaN();
        std::get<2>(optimum) = std::numeric_limits<scalar_t>::max();
        for (int step = 0; step < steps; ++ step)
        {
            for (const auto param1 : space1.generate(maximum_trials_per_step))
            {
                for (const auto param2 : space2.generate(maximum_trials_per_step))
                {
                    const auto params = std::make_tuple(param1, param2);
                    if (!detail::checked(tried, params))
                    {
                        detail::update(optimum, params, evaluator(param1, param2));
                    }
                }
            }

            if (step + 1 < steps)
            {
                space1.refine(std::get<0>(optimum));
                space2.refine(std::get<1>(optimum));
            }
        }

        return optimum;
    }

    ///
    /// \brief coarse-to-fine tuning of three continuous hyper-parameters.
    ///     the tuning is performed in steps by sampling with finer and finer step size around the current optimum.
    ///
    /// NB: the evaluator is called with (param1, param2, param3) and returns a tuple with:
    ///         <value_for_param1_2_3, others_for_param1_2_3...>
    /// NB: the final returned value is a tuple with:
    ///         <optimum_param1, optimum_param2, optimum_param3, value_for_optimum_param1_2_3, others_for_optimum_param1_2_3...>
    ///
    template <typename tspace1, typename tspace2, typename tspace3, typename tevaluator>
    auto grid_tune(tspace1 space1, tspace2 space2, tspace3 space3,
        const tevaluator& evaluator, const int maximum_trials_per_step, const int steps)
    {
        assert(steps > 0 && maximum_trials_per_step > 3);

        std::vector<std::tuple<scalar_t, scalar_t, scalar_t>> tried;

        decltype(std::tuple_cat(std::make_tuple(scalar_t{}, scalar_t{}, scalar_t{}), evaluator(0, 0, 0))) optimum;
        std::get<0>(optimum) = std::numeric_limits<scalar_t>::quiet_NaN();
        std::get<1>(optimum) = std::numeric_limits<scalar_t>::quiet_NaN();
        std::get<2>(optimum) = std::numeric_limits<scalar_t>::quiet_NaN();
        std::get<3>(optimum) = std::numeric_limits<scalar_t>::max();
        for (int step = 0; step < steps; ++ step)
        {
            for (const auto param1 : space1.generate(maximum_trials_per_step))
            {
                for (const auto param2 : space2.generate(maximum_trials_per_step))
                {
                    for (const auto param3 : space3.generate(maximum_trials_per_step))
                    {
                        const auto params = std::make_tuple(param1, param2, param3);
                        if (!detail::checked(tried, params))
                        {
                            detail::update(optimum, params, evaluator(param1, param2, param3));
                        }
                    }
                }
            }

            if (step + 1 < steps)
            {
                space1.refine(std::get<0>(optimum));
                space2.refine(std::get<1>(optimum));
                space3.refine(std::get<2>(optimum));
            }
        }

        return optimum;
    }
}
