#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief gradient descent with line-search.
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_gd_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_gd_t();

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
