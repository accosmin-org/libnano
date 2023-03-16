#pragma once

#include <nano/solver/state.h>

namespace nano
{
    ///
    /// \brief wrapper over the solver state to decide convergence for non-smooth optimization problems
    ///     if no significant improvement in the given number of last iterations (aka patience).
    ///
    class nonsmooth_solver_state_t
    {
    public:
        ///
        /// \brief constructor
        ///
        nonsmooth_solver_state_t(solver_state_t& state, tensor_size_t patience);

        ///
        /// \brief update the current state, if the given function value is smaller than the current one.
        /// returns true if the update was performed.
        ///
        bool update_if_better(const vector_t& x, scalar_t fx);
        bool update_if_better(const vector_t& x, const vector_t& gx, scalar_t fx);

        ///
        /// \brief returns true if convergence is detected.
        ///
        bool converged(scalar_t epsilon) const;

    private:
        // attributes
        solver_state_t& m_state;
        tensor_size_t   m_iteration{0};
        vector_t        m_df_history;
        vector_t        m_dx_history;
    };
} // namespace nano
