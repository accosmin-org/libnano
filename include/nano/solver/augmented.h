#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief (practical) augmented lagrangian method.
    ///
    /// see "Practical Augmented Lagrangian Methods", by E. G. Birgin, J. M. Martinez, 2007.
    ///
    /// NB: the penalty method works by increasing the penalty term in the outer loop
    ///     and using the given solver to minimize the penalty function (the inner loop).
    ///
    /// NB: the penalty function is exact,
    ///     but it is not smooth and thus the optimization is typically not very precise.
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
