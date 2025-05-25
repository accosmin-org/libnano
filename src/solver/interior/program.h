#pragma once

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
/// NB: the implementation follows the notation from (2).
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
///         x'           = |x y|^T
///         G' * x' - h' = -y
///
class program_t
{
public:
    // FIXME: make this an option of the interior point solver
    enum class scale_type : uint8_t
    {
        ruiz,
        none,
    };

    program_t(const linear_program_t&, linear_constraints_t, scale_type, scalar_t miu);

    program_t(const quadratic_program_t&, linear_constraints_t, scale_type, scalar_t miu);

    const vector_t& x() const { return m_x; }

    const vector_t& u() const { return m_u; }

    const vector_t& v() const { return m_v; }

    const vector_t& dx() const { return m_dx; }

    const vector_t& du() const { return m_du; }

    const vector_t& dv() const { return m_dv; }

    const vector_t& original_x() const { return m_orig_x; }

    const vector_t& original_u() const { return m_orig_u; }

    const vector_t& original_v() const { return m_orig_v; }

    const function_t& function() const { return m_function; }

    ///
    /// \brief compute the residual for the current state (x, u, v).
    ///
    scalar_t residual() const;

    ///
    /// \brief update and return the residual for the trial (x + xstep * dx, u + ustep * du, v + vstep * dv).
    ///
    scalar_t update(scalar_t xstep, scalar_t ustep, scalar_t vstep, scalar_t miu, bool apply = false);

    ///
    /// \brief return the maximum step that can be take in the direction (x, x + xstep * dx) that keeps y > 0.
    ///
    scalar_t max_xstep() const;

    ///
    /// \brief return the maximum step that can be take in the direction (u, u + ustep * du) that keeps u > 0.
    ///
    scalar_t max_ustep() const;

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
    program_t(const function_t&, matrix_t Q, vector_t c, linear_constraints_t, scale_type, scalar_t miu);

    void update_original();

    tensor_size_t n() const { return m_c.size(); }

    tensor_size_t p() const { return m_A.rows(); }

    tensor_size_t m() const { return m_G.rows(); }

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
};
} // namespace nano
