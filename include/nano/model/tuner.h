#pragma once

#include <nano/solver/state.h>
#include <nano/core/estimator.h>
#include <nano/model/param_space.h>

namespace nano
{
    ///
    /// \brief utility to tune (hyper-)parameters by fitting and minimizing iteratively
    ///     a quadratic surrogate function that maps parameters to a scalar value function
    ///     (the lower, the better).
    ///
    class NANO_PUBLIC tuner_t : public estimator_t
    {
    public:

        ///< callback(parameter values) returns the associated value
        using callback_t = std::function<scalar_t(const tensor1d_t&)>;

        ///< optimization step
        struct step_t
        {
            step_t();
            step_t(tensor1d_t param, const callback_t&);
            step_t(tensor1d_t param, solver_state_t surrogate_fit, solver_state_t surrogate_opt, const callback_t&);

             static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

            tensor1d_t      m_param, m_opt_param;   ///< parameter values tested at this step and the optimum so far
            scalar_t        m_value{NaN}, m_opt_value{NaN};   ///< current value and the optimum so far
            solver_state_t  m_surrogate_fit;        ///< fitted surrogate function to the data (optional)
            solver_state_t  m_surrogate_opt;        ///< optimum of the fitted surrogate function (optional)
        };
        using steps_t = std::vector<step_t>;

        ///
        /// \brief constructor
        ///
        explicit tuner_t(param_spaces_t, callback_t);

        ///
        /// \brief tune the (hyper-)parameters starting from the given initial parameter values.
        ///
        steps_t optimize(const tensor2d_t& initial_params) const;

    private:

        // attributes
        param_spaces_t  m_param_spaces;         ///<
        callback_t      m_callback;             ///<
    };
}
