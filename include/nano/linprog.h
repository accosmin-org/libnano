#pragma once

#include <memory>
#include <nano/configurable.h>
#include <nano/eigen.h>

namespace nano::linprog
{
///
/// \brief the standard form of linear programming:
///     f(x) = c.dot(x) s.t Ax = b and x >= 0.
///
/// see (1) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
/// see (2) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
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
/// \brief the solution of the standard form of linear programming.
///
struct NANO_PUBLIC solution_t
{
    ///
    /// \brief returns true if convergence is detected.
    ///
    bool converged(scalar_t max_kkt_violation = 1e-16) const;

    // attributes
    static constexpr auto max = std::numeric_limits<scalar_t>::max();

    vector_t m_x;        ///< solution (primal problem)
    vector_t m_l;        ///< solution (dual problem) - equality constraints
    vector_t m_s;        ///< solution (dual problem) - inequality constraints
    int      m_iters{0}; ///< number of iterations
    scalar_t m_miu{max}; ///< duality measure: ~zero (converged), very large/infinite (unfeasible/unbounded)
    scalar_t m_kkt{max}; ///< deviation of KKT conditions: ~zero (converged), very large/infinite (unfeasible/unbounded)
    scalar_t m_ldlt_rcond{0};        ///< LDLT decomposition: reciprocal condition number
    bool     m_ldlt_positive{false}; ///< LDLT decomposition: positive semidefinite?! (if not, unstable system)
};

///
/// \brief the inequality form of linear programming:
///     f(x) = c.dot(x) s.t Ax <= b.
///
/// see (1) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
/// see (2) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
///
struct NANO_PUBLIC inequality_problem_t
{
    ///
    /// \brief constructor
    ///
    inequality_problem_t(vector_t c, matrix_t A, vector_t b);

    ///
    /// \brief return the equivalent standard form problem.
    ///
    problem_t transform() const;

    ///
    /// \brief return the equivalent solution from the given solution of the equivalent standard form problem.
    ///
    solution_t transform(const solution_t&) const;

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
/// \brief the general form of linear programming:
///     f(x) = c.dot(x) s.t Ax = b and Gx <= h.
///
/// see (1) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
/// see (2) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
///
struct NANO_PUBLIC general_problem_t
{
    ///
    /// \brief constructor
    ///
    general_problem_t(vector_t c, matrix_t A, vector_t b, matrix_t G, vector_t h);

    ///
    /// \brief return the equivalent standard form problem.
    ///
    problem_t transform() const;

    ///
    /// \brief return the equivalent solution from the given solution of the equivalent standard form problem.
    ///
    solution_t transform(const solution_t&) const;

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(const vector_t& x, scalar_t epsilon) const;

    // attributes
    vector_t m_c; ///<
    matrix_t m_A; ///<
    vector_t m_b; ///<
    matrix_t m_G; ///<
    vector_t m_h; ///<
};

///
/// \brief solver for linear programming problems.
///
/// see (1) "On the implementation of a primal-dual interior point method", by S. Mehrotra, 1992.
/// see (2) ch.14 (page 411) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
/// NB: the parameter `eta` is implemented as:
///     `eta_k = 1 - eta0 / (1 + k)^etaP`, where k is the current iteration index
///     to converge to 1 so that the algorithm converges fast.
///
/// NB: the faster `eta_k` approaches 1, the faster the convergence.
/// NB: the solution is found with a 1e-8 accuracy in general in less than 10 iterations with the default settings.
/// NB: more accurate solutions are obtained by decreasing the convergence speed (to 1) of `eta_k` at a cost of
///     higher number of iterations. this can be achieved with setting `etaP` to either 1 or 2.
///
class NANO_PUBLIC solver_t final : public configurable_t
{
public:
    ///
    /// \brief logging operator: op(proble, solution).
    ///
    using logger_t = std::function<void(const problem_t&, const solution_t&)>;

    ///
    /// \brief constructor
    ///
    explicit solver_t(logger_t logger = logger_t{});

    ///
    /// \brief returns the solution of the given linear program using the predictor-corrector algorithm.
    ///
    solution_t solve(const problem_t&) const;
    solution_t solve(const general_problem_t&) const;
    solution_t solve(const inequality_problem_t&) const;

private:
    solution_t solve_(const problem_t&) const;

    // attributes
    logger_t m_logger; ///<
};

///
/// \brief check if the equality constraints `Ax = b` are full row rank
///     and if so return the row-independant linear constraints by performing an appropriate matrix decomposition.
///
NANO_PUBLIC std::optional<std::pair<matrix_t, vector_t>> make_independant_equality_constraints(const matrix_t& A,
                                                                                               const vector_t& b);
} // namespace nano::linprog
