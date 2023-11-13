#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief parameter-free stochastic gradient method using coin betting strategies.
///
/// see "Training Deep Networks without Learning Rates through Coin Betting", by F. Orabona, T. Tommasi, 2017
///
/// NB: the functional constraints (if any) are all ignored.
/// NB: the algorithm is very slow and it is provided as a baseline.
/// NB: the range of the gradients is updated on the fly (see the COCOB-Backprop variation).
/// NB: the iterations are stopped when no significant decrease in the function value in the recent iterations.
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
