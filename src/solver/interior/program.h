#pragma once

#include <nano/function/linear.h>
#include <nano/function/quadratic.h>
#include <nano/function/util.h>

namespace nano
{
///
/// \brief solver for the the linear system of equations resulting from the KKT conditions solved with the primal-dual
/// interior point method applied
///
/// to the linear program:
///     min  c.dot(x)
///     s.t. G * x <= h
///          A * x = b
///
/// or to the quadratic program:
///
///     min  0.5 * x.dot(Q * x) + c.dot(x)
///     s.t. G * x <= h
///          A * x = b
///
/// see (1) ch.5,6 "Primal-dual interior-point methods", by S. Wright, 1997.
/// see (2) ch.11 "Convex Optimization", by S. Boyd and L. Vandenberghe, 2004.
/// see (3) ch.14,16,19 "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
/// NB: the implementation follows the notation from (2).
///
class program_t
{
public:
    program_t(const linear_program_t&, linear_constraints_t);

    program_t(const quadratic_program_t&, linear_constraints_t);

    const matrix_t& Q() const { return m_Q; }

    const vector_t& c() const { return m_c; }

    const matrix_t& A() const { return m_A; }

    const matrix_t& G() const { return m_G; }

    const vector_t& b() const { return m_b; }

    const vector_t& h() const { return m_h; }

    const vector_t& x() const { return m_x; }

    const vector_t& u() const { return m_u; }

    const vector_t& v() const { return m_v; }

    const vector_t& dx() const { return m_dx; }

    const vector_t& du() const { return m_du; }

    const vector_t& dv() const { return m_dv; }

    tensor_size_t n() const { return m_c.size(); }

    tensor_size_t p() const { return m_A.rows(); }

    tensor_size_t m() const { return m_G.rows(); }

    const function_t& function() const { return m_function; }

    ///
    /// \brief compute the KKT optimality test for the current state (x, u, v).
    ///
    scalar_t kkt_test() const;

    ///
    /// \brief compute the KKT optimality test for the trial line-search state
    ///     (x + xstep * dx, u + ustep * du, v + vstep * dv).
    ///
    scalar_t kkt_test(scalar_t xstep, scalar_t ustep, scalar_t vstep) const;

    ///
    /// \brief change state to (x0, u0, v0) and update residuals.
    ///
    void update(vector_t x0, vector_t u0, vector_t v0, scalar_t miu);

    ///
    /// \brief change state to the result of the line-search (x + xstep * dx, u + ustep * du, v + vstep * dv) and update
    /// residuals.
    ///
    void update(scalar_t xstep, scalar_t ustep, scalar_t vstep, scalar_t miu);

    ///
    /// \brief compute the state update (dx, du, dv) by solving the linear system of equations derived from the KKT
    /// conditions.
    ///
    struct solve_stats_t
    {
        bool     m_valid{false};   ///< indicates if the updates are finite
        scalar_t m_precision{0.0}; ///< precision with which the linear system of equations was solved
    };

    solve_stats_t solve();

private:
    program_t(const function_t&, matrix_t Q, vector_t c, linear_constraints_t);

    solve_stats_t solve_noA();
    solve_stats_t solve_noG();
    solve_stats_t solve_wAG();

    template <class tx, class tu, class tv>
    scalar_t kkt_optimality_test(const tx& x, const tu& u, const tv& v) const
    {
        const auto m = this->m();
        const auto p = this->p();

        const auto Gxmh = m_G * x - m_h;
        const auto Axmb = m_A * x - m_b;
        const auto Atv  = m_A.transpose() * v;
        const auto Gtu  = m_G.transpose() * u;

        const auto kkt1 = (m == 0) ? 0.0 : Gxmh.array().max(0.0).matrix().template lpNorm<Eigen::Infinity>();
        const auto kkt2 = (p == 0) ? 0.0 : Axmb.template lpNorm<Eigen::Infinity>();
        const auto kkt3 = (m == 0) ? 0.0 : (-u.array()).max(0.0).matrix().template lpNorm<Eigen::Infinity>();
        const auto kkt4 = (m == 0) ? 0.0 : (u.array() * Gxmh.array()).matrix().template lpNorm<Eigen::Infinity>();
        const auto kkt5 = (m_Q.size() > 0) ? (m_Q * x + m_c + Atv + Gtu).template lpNorm<Eigen::Infinity>()
                                           : (m_c + Atv + Gtu).template lpNorm<Eigen::Infinity>();

        return std::max({kkt1, kkt2, kkt3, kkt4, kkt5});
    }

    // attributes
    const function_t&    m_function; ///< original function to minimize
    matrix_t             m_Q;        ///< objective: 1/2 * x.dot(Q * x) + c.dot(x)
    vector_t             m_c;        ///<
    matrix_t             m_A;        ///< equality constraints: A * x = b
    vector_t             m_b;        ///<
    matrix_t             m_G;        ///< inequality constraints: G * x <= h
    vector_t             m_h;        ///<
    vector_t             m_x;        ///< solution
    vector_t             m_u;        ///< Lagrange multipliers for the inequality constraints
    vector_t             m_v;        ///< Lagrange multipliers for the equality constraints
    vector_t             m_dx;       ///< current variation of the solution
    vector_t             m_du;       ///< current variation of Lagrange multipliers for the inequality constraints
    vector_t             m_dv;       ///< current variation of Lagrange multipliers for the equality constraints
    vector_t             m_rdual;    ///< dual residual
    vector_t             m_rcent;    ///< centering residual
    vector_t             m_rprim;    ///< primal residual
};
} // namespace nano
