#pragma once

#include <nano/solver/status.h>
#include <nano/tensor.h>

namespace nano::program
{
///
/// \brief the state of a primal-dual interior-point solver.
///
/// NB: the KKT optimality test as the maximum of the infinite norm of the 5 vector conditions from:
/// see (1) ch.5 "Convex Optimization", by S. Boyd and L. Vandenberghe, 2004.
///
/// test 1: g_i(x) <= 0 (inequalities satisfied)
/// test 2: h_j(x) == 0 (equalities satisfied)
/// test 3: lambda_i >= 0 (positive multipliers for the inequalities)
/// test 4: lambda_i * g_i(x) == 0
/// test 5: grad(f(x)) + sum(lambda_i * grad(g_i(x))) + sum(miu_j * h_j(x)) == 0
///
struct NANO_PUBLIC solver_state_t
{
    ///
    /// \brief default constructor
    ///
    solver_state_t();

    ///
    /// \brief constructor
    ///
    solver_state_t(tensor_size_t n, tensor_size_t n_ineqs, tensor_size_t n_eqs);

    ///
    /// \brief return the cumulated residual.
    ///
    scalar_t residual() const;

    ///
    /// \brief compute and store the KKT optimality test for the given linear/quadratic program.
    ///
    void update(const matrix_t& Q, const vector_t& c, const matrix_t& A, const vector_t& b, const matrix_t& G,
                const vector_t& h);

    static constexpr auto max = std::numeric_limits<scalar_t>::max();
    static constexpr auto nan = std::numeric_limits<scalar_t>::quiet_NaN();

    // attributes
    int           m_iters{0};                         ///< number of iterations
    scalar_t      m_fx{nan};                          ///< objective
    vector_t      m_x;                                ///< solution (primal problem)
    vector_t      m_u;                                ///< Lagrange multipliers (inequality constraints)
    vector_t      m_v;                                ///< Lagrange multipliers (equality constraints)
    scalar_t      m_eta{nan};                         ///< surrogate duality gap
    vector_t      m_rdual;                            ///< dual residual
    vector_t      m_rcent;                            ///< central residual
    vector_t      m_rprim;                            ///< primal residual
    scalar_t      m_kkt{0};                           ///< KKT optimality test
    solver_status m_status{solver_status::max_iters}; ///< optimization status
    scalar_t      m_ldlt_rcond{0};                    ///< LDLT decomp: reciprocal condition number
    bool          m_ldlt_positive{false};             ///< LDLT decomp: positive semidefinite?, otherwise unstable
};

///
/// \brief pretty print the given solver state.
///
NANO_PUBLIC std::ostream& operator<<(std::ostream&, const solver_state_t&);
} // namespace nano::program
