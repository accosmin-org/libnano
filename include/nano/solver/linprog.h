#pragma once

#include <nano/arch.h>
#include <nano/eigen.h>

namespace nano::linprog
{
///
/// \brief the standard form of linear programming:
///     f(x) = c.dot(x) s.t Ax = b and x >= 0,
///
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
struct NANO_PUBLIC problem_t
{
    ///
    /// \brief constructor
    ///
    problem_t(vector_t c, matrix_t A, vector_t b);

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(const vector_t& x, scalar_t epsilon) const;

    // attributes
    vector_t m_c; ///<
    matrix_t m_A; ///<
    vector_t m_b; ///<
};

///
/// \brief the solution to the standard form of linear programming.
///
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
struct NANO_PUBLIC solution_t
{
    ///
    /// \brief returns true if convergence is detected.
    ///
    bool converged(scalar_t max_duality_measure = 1e-14) const;

    ///
    /// \brief returns true if convergence is detected (not feasible or unbounded problem).
    ///
    bool diverged(scalar_t min_duality_measure = 1e+12) const;

    // attributes
    vector_t m_x;        ///< solution (primal problem)
    vector_t m_l;        ///< solution (dual problem) - equality constraints
    vector_t m_s;        ///< solution (dual problem) - inequality constraints
    int      m_iters{0}; ///< number of iterations
    scalar_t m_miu{0.0}; ///< duality measure: ~zero (converged), very large/infinite (not feasible, unbounded)
};

using logger_t = std::function<void(const solution_t&)>;

///
/// \brief return a starting point appropriate for primal-dual interior point methods.
///
/// see ch.14 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
NANO_PUBLIC solution_t make_starting_point(const problem_t&);

///
/// \brief returns the solution of the given linear program using the predictor-corrector algorithm.
///
/// see (1) "On the implementation of a primal-dual interior point method", by S. Mehrotra, 1992.
/// see (2) ch.14 (page 411) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
NANO_PUBLIC solution_t solve(const problem_t&, const logger_t& logger = logger_t{});
} // namespace nano::linprog
