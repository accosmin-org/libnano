#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief a bundle algorithm for nonsmooth convex problems that combines the proximal and the level methods.
///
/// see (1) "A doubly stabilized bundle method for nonsmooth convex optimization", by Oliveira, Solodov, 2013
///
class NANO_PUBLIC solver_dsbm_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit solver_dsbm_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0, const logger_t&) const override;
};
} // namespace nano
