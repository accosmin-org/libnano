#pragma once

#include <nano/configurable.h>
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
class NANO_PUBLIC solver_t final : public configurable_t
{
public:
    ///
    /// \brief logging operator: op(state), returns false if the optimization should stop.
    ///
    using logger_t = std::function<bool(const solver_state_t&)>;

    ///
    /// \brief constructor
    ///
    explicit solver_t(logger_t logger = logger_t{});

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
    bool log(const solver_state_t&) const;
    void done(solver_state_t&, scalar_t epsilon) const;

    struct program_t;

    solver_state_t solve_without_inequality(const program_t&) const;
    solver_state_t solve_with_inequality(const program_t&, const vector_t&) const;

    // attributes
    logger_t m_logger; ///<
};
} // namespace nano::program
