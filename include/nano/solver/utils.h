#pragma once

#include <nano/solver.h>

namespace nano::constrained
{
    inline bool converged(const solver_state_t& bstate, const solver_state_t& cstate, const scalar_t epsilon)
    {
        const auto dx = (cstate.x - bstate.x).lpNorm<Eigen::Infinity>();
        const auto df = std::fabs(cstate.f - bstate.f);

        return cstate.constraint_test() < epsilon &&
               (dx < epsilon * std::max(1.0, bstate.x.lpNorm<Eigen::Infinity>())) &&
               (df < epsilon * std::max(1.0, std::fabs(bstate.f)));
    }

    inline void more_precise(const rsolver_t& solver, const scalar_t epsilonK)
    {
        solver->parameter("solver::epsilon") = solver->parameter("solver::epsilon").value<scalar_t>() * epsilonK;
    }
} // namespace nano::constrained
