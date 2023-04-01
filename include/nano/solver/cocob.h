#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief parameter-free stochastic gradient method using coin betting strategies.
    ///     see "Training Deep Networks without Learning Rates through Coin Betting", by F. Orabona, T. Tommasi, 2017
    ///
    /// NB: the functional constraints (if any) are all ignored.
    /// NB: the convergence criterion is that the difference of two consecutive best updates is smaller than epsilon.
    ///
    class NANO_PUBLIC solver_cocob_t final : public solver_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        solver_cocob_t();

        ///
        /// \brief @see clonable_t
        ///
        rsolver_t clone() const override;

        ///
        /// \brief @see solver_t
        ///
        solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;
    };
} // namespace nano
