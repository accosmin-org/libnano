#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief (practical) augmented lagrangian method.
    ///
    /// see "Practical Augmented Lagrangian Methods", by E. G. Birgin, J. M. Martinez, 2007.
    ///
    /// NB: the penalty parameter and the estimated lagrangian multipliers are updated at the same time
    ///     in a loop to solve a potentially smooth function - the augmented lagrangian
    ///     (continuous and differentiable, but not necessarily with continuous gradients).
    ///
    /// NB: the augmented lagrangian is solved without any bounds constraints like in the original formulation.
    ///
    class NANO_PUBLIC solver_augmented_lagrangian_t final : public solver_t
    {
    public:
        ///
        /// \brief constructor
        ///
        solver_augmented_lagrangian_t();

        ///
        /// \brief @see clonable_t
        ///
        rsolver_t clone() const final;

        ///
        /// \brief @see solver_t
        ///
        solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;
    };
} // namespace nano