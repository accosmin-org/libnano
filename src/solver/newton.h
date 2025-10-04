#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief (truncated) newton method with line-search.
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_newton_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_newton_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    bool has_lsearch() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0, const logger_t&) const override;
};
} // namespace nano
