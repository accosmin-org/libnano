#pragma once

#include <nano/solver.h>

namespace nano::constrained
{
    inline bool converged(const solver_state_t& bstate, const solver_state_t& cstate, const scalar_t epsilon)
    {
        return cstate.constraint_test() < epsilon && (cstate.x - bstate.x).lpNorm<Eigen::Infinity>() < epsilon;
    }

    inline void more_precise(const rsolver_t& solver, const scalar_t epsilonK)
    {
        solver->parameter("solver::epsilon") = solver->parameter("solver::epsilon").value<scalar_t>() * epsilonK;
    }
} // namespace nano::constrained
