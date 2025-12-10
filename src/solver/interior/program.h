#pragma once

#include <Eigen/Dense>
#include <nano/function/linear.h>
#include <nano/function/quadratic.h>
#include <nano/function/util.h>

namespace nano
{
///
/// \brief models the linear equations following produced by the primal-dual interior point method applied
///
/// to the linear program:
///     min. c.dot(x)
///     s.t. G * x <= h
///          A * x = b
///
/// or to the quadratic program:
///
///     min. 0.5 * x.dot(Q * x) + c.dot(x)
///     s.t. G * x <= h
///          A * x = b
///
/// see (1) ch.5,6 "Primal-dual interior-point methods", by S. Wright, 1997.
/// see (2) ch.11 "Convex Optimization", by S. Boyd and L. Vandenberghe, 2004.
/// see (3) ch.14,16,19 "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
/// NB: the implementation follows the notation from (2, ch. 11).
/// NB: the original tensors (Q, c, G, h, A, b) are scaled in-place.
/// NB: the primal-dual iterate (x, u, v) are also scaled similarly.
///
/// NB: internally the inequality constraints `G * x <= h` are transformed to `G * x + y = h` and `-y <= 0`
///     (so that a strictly feasible starting point is easily produced, e.g. `y = 1.0`).
///
///     the matrices for the transformed quadratic program are:
///         Q' = |Q 0|,  c' = |c|
///              |0 0|        |0|
///
///         A' = |A 0|,  b' = |b|
///              |G I|        |h|
///
///         G' = |0 -I|, h' = 0
///
///     with:
///         x'        = |x y|^T
///         G * x - h = -y
///
///     the primal variables are thus (x, y), while
///     the dual variables are (u for -y <= 0, v for A * x = b, w for G * x + y = h).
///
class program_t
{
public:
    program_t(const linear_program_t&, linear_constraints_t, const vector_t& x0);

    program_t(const quadratic_program_t&, linear_constraints_t, const vector_t& x0);

    const vector_t& original_x() const { return m_orig_x; }

    const vector_t& original_u() const { return m_orig_u; }

    const vector_t& original_v() const { return m_orig_v; }

    const function_t& function() const { return m_function; }

    ///
    /// \brief update the primal-dual variables (x, y, u, v, w) following
    ///     the predictor-corrector algorithm 16.4 from (3).
    ///
    struct kkt_stats_t
    {
        scalar_t m_precision{0.0};  ///< precision with which the linear system of equations was solved
        scalar_t m_rcond{0.0};      ///< estimation of the reciprocal condition of the matrix to decompose
        bool     m_valid{false};    ///< indicates if the updates are finite
        bool     m_positive{false}; ///< is the original matrix positive semidefinite?
        bool     m_negative{false}; ///< is the original matrix negative semidefinite?
    };

    struct stats_t
    {
        bool        m_valid{false};         ///<
        scalar_t    m_alpha{0.0};           ///< step length (primal/dual)
        scalar_t    m_sigma{0.0};           ///< centering parameter
        scalar_t    m_primal_residual{0.0}; ///<
        scalar_t    m_dual_residual{0.0};   ///<
        scalar_t    m_duality_gap{0.0};     ///<
        kkt_stats_t m_predictor_stats;      ///< statistics for solving the KKT system (predictor step)
        kkt_stats_t m_corrector_stats;      ///< statistics for solving the KKT system (corrector step)
    };

    stats_t update(scalar_t tau);

private:
    program_t(const function_t&, matrix_t Q, vector_t c, linear_constraints_t, const vector_t& x0);

    kkt_stats_t solve();

    void update_solver();
    void update_original();
    void update_residual(scalar_t sigma);

    tensor_size_t n() const { return m_c.size(); }

    tensor_size_t p() const { return m_A.rows(); }

    tensor_size_t m() const { return m_G.rows(); }

    auto unpack_dims() const { return std::make_tuple(n(), m(), p()); }

    auto unpack_vars()
    {
        const auto [n, m, p] = unpack_dims();

        return std::make_tuple(m_x.segment(0, n), m_x.segment(n, m), m_u.segment(0, m), m_v.segment(0, p),
                               m_v.segment(p, m));
    }

    auto unpack_delta()
    {
        const auto [n, m, p] = unpack_dims();

        return std::make_tuple(m_dx.segment(0, n), m_dx.segment(n, m), m_du.segment(0, m), m_dv.segment(0, p),
                               m_dv.segment(p, m));
    }

    using kkt_solver_t = Eigen::LDLT<eigen_matrix_t<scalar_t>>;

    // attributes
    const function_t& m_function; ///< original function to minimize
    matrix_t          m_Q;        ///< objective: 1/2 * x.dot(Q * x) + c.dot(x)
    vector_t          m_c;        ///<
    matrix_t          m_G;        ///< inequality constraints: G * x <= h
    vector_t          m_h;        ///<
    matrix_t          m_A;        ///< equality constraints: A * x = b
    vector_t          m_b;        ///<
    vector_t          m_x;        ///< solution
    vector_t          m_u;        ///< Lagrange multipliers for the inequality constraints
    vector_t          m_v;        ///< Lagrange multipliers for the equality constraints
    vector_t          m_dx;       ///< current variation of the solution
    vector_t          m_du;       ///< current variation of Lagrange multipliers for the inequality constraints
    vector_t          m_dv;       ///< current variation of Lagrange multipliers for the equality constraints
    vector_t          m_dQ;       ///<
    vector_t          m_dG;       ///<
    vector_t          m_dA;       ///<
    vector_t          m_rdual;    ///< dual residual
    vector_t          m_rcent;    ///< centering residual
    vector_t          m_rprim;    ///< primal residual
    vector_t          m_orig_x;   ///< scaled solution
    vector_t          m_orig_u;   ///< scaled Lagrange multipliers for the inequality constraints
    vector_t          m_orig_v;   ///< scaled Lagrange multipliers for the equality constraints
    matrix_t          m_lmat;     ///< reduced KKT system: lmat * lsol = lvec
    vector_t          m_lvec;     ///<
    vector_t          m_lsol;     ///<
    kkt_solver_t      m_solver;   ///<
};
} // namespace nano
