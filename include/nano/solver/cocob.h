#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief parameter-free stochastic gradient method using coin betting strategies.
    ///     see "Training Deep Networks without Learning Rates through Coin Betting", by F. Orabona, T. Tommasi, 2017
    ///
    class NANO_PUBLIC solver_cocob_t final : public solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cocob_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;
    };
}
