#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief limited memory BGFS (l-BGFS).
///
/// see "Updating Quasi-Newton Matrices with Limited Storage", by J. Nocedal, 1980
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_lbfgs_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_lbfgs_t();

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
