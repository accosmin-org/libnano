#pragma once

#include <nano/configurable.h>
#include <nano/loggable.h>
#include <nano/program/linear.h>
#include <nano/program/quadratic.h>
#include <nano/program/state.h>

namespace nano::program
{
///
/// \brief primal-dual interior-point solver specialized for linear and quadratic programs.
///
/// see (1) ch.5,6 "Primal-dual interior-point methods", by S. Wright, 1997.
/// see (2) ch.11 "Convex Optimization", by S. Boyd and L. Vandenberghe, 2004.
/// see (3) ch.14,16,19 "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
class NANO_PUBLIC solver_t final : public configurable_t, public loggable_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_t();

    ///
    /// \brief returns the solution of the given linear program.
    ///
    solver_state_t solve(const linear_program_t&) const;
    solver_state_t solve(const linear_program_t&, const vector_t& x0) const;

    ///
    /// \brief returns the solution of the given quadratic program.
    ///
    solver_state_t solve(const quadratic_program_t&) const;
    solver_state_t solve(const quadratic_program_t&, const vector_t& x0) const;

private:
    struct program_t;

    void done(const program_t&, solver_state_t&, scalar_t epsilon) const;

    solver_state_t solve_without_inequality(const program_t&) const;
    solver_state_t solve_with_inequality(const program_t&, const vector_t&) const;
};
} // namespace nano::program
