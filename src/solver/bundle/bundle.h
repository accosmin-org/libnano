#pragma once

#include <nano/function/quadratic.h>
#include <nano/solver.h>
#include <nano/tensor/algorithm.h>

namespace nano
{
///
/// \brief models the bundle of sub-gradients as used by penalized (proximal) bundle algorithms.
///
/// see (1) "A doubly stabilized bundle method for nonsmooth convex optimization", by Oliveira, Solodov, 2013
/// see (2) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
/// see (3) "Variable metric bundle methods: from conceptual to implementable forms", by Lemarechal, Sagastizabal, 1997
/// see (4) "Dynamical adjustment of the prox-parameter in bundle methods", by Rey, Sagastizabal, 2002
/// see (5) "Fast proximal algorithms for nonsmooth convex optimization", by Ouorou, 2020
/// see (6) "A NU-algorithm for convex minimization", by Mifflin, Sagastizabal, 2005
///
/// NB: the implementation follows the notation from (1). if the level parameter is infinite,
///     then this formulation becames the penalized proximal bundle algorithms
///     (see 2, RQB from 3, mRQB from 4, FPBA1/FPBA2 from 5).
///
/// NB: the bundle is kept small by:
///     - first removing all inactive constraints and
///     - then the ones with the smallest Lagrange multipliers if needed - see (1, ch 5.1.4).
///
class NANO_PUBLIC bundle_t
{
public:
    ///
    /// \brief solution to the quadratic optimization problem from (1).
    ///
    struct solution_t
    {
        explicit solution_t(tensor_size_t dims = 0);

        // attributes
        vector_t m_x;            ///< optimum: stability center
        scalar_t m_r{0.0};       ///< optimum: level (if applicable)
        scalar_t m_tau{0.0};     ///< proximal parameter
        vector_t m_alphas;       ///< Lagrangian multiplier associated to the bundle inequalities
        scalar_t m_lambda{0.0};  ///< Lagrangian multiplier associated to the level inequality (if applicable)
        bool     m_valid{false}; ///< indicates if the post-conditions of the solution are satisfied
    };

    ///
    /// \brief constructor
    ///
    bundle_t(const solver_state_t&, tensor_size_t max_size);

    ///
    /// \brief setup the default configuration.
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief construct an empty bundle with the given configuration.
    ///
    static bundle_t make(const solver_state_t&, const configurable_t&, const string_t& prefix);

    ///
    /// \brief return the current size of the bundle.
    ///
    tensor_size_t size() const { return m_bsize; }

    ///
    /// \brief return the stability center.
    ///
    const vector_t& x() const { return m_x; }

    ///
    /// \brief return the sub-gradient at the stability center.
    ///
    const vector_t& gx() const { return m_gx; }

    ///
    /// \brief return the function value at the stability center.
    ///
    scalar_t fx() const { return m_fx; }

    ///
    /// \brief evaluate the cutting plange model at the given point
    ///     (maximum over the linearizations in the bundle).
    ///
    scalar_t fhat(const vector_t& x) const;

    ///
    /// \brief return the tolerance for error-like statistics, see (1):
    ///     error/delta/ehat <= epsilon * (1 + |f(x_k)|).
    ///
    scalar_t etol(scalar_t epsilon) const;

    ///
    /// \brief return the tolerance for the smeared gradient:
    ///     |G_hat| <= epsilon * 1e+2 * (1 + |f(x_k)|).
    ///
    /// NB: this is different from any of the given references as (3) doesn't use a specific criterion,
    ///     while (1) uses `epsilon * sqrt(n)` which doesn't work for badly scaled problems.
    ///
    scalar_t gtol(scalar_t epsilon) const;

    ///
    /// \brief change the proximity center to the given point and update the bundle.
    ///
    void moveto(vector_cmap_t y, vector_cmap_t gy, scalar_t fy);

    ///
    /// \brief update the bundle with the given point.
    ///
    void append(vector_cmap_t y, vector_cmap_t gy, scalar_t fy);

    ///
    /// \brief return the solution of the doubly stabilized bundle problem (1):
    ///     argmin_(x, r) r + 1/(2 * tau) * (x - x_k^).dot(M * (x - x_k^))
    ///             s.t.  f_j + <g_j, x - x_j> <= r (for all sub-gradients j in the bundle)
    ///             s.t.  r <= l_k (the level parameter).
    ///
    ///     where x_k^ is the current proximal stability center and
    ///           M is a symmetric positive-definite matrix.
    ///
    /// NB: a convex quadratic program.
    /// NB: M is the identity matrix in (1).
    ///
    const solution_t& solve(scalar_t tau, scalar_t level, const logger_t&);
    const solution_t& solve(const matrix_t& M, scalar_t tau, scalar_t level, const logger_t&);

private:
    tensor_size_t dims() const { return m_x.size(); }

    tensor_size_t capacity() const { return m_bundleH.size(); }

    template <class toperator>
    tensor_size_t remove_if(const toperator& op)
    {
        auto bundleG = m_bundleG.slice(0, m_bsize);
        auto bundleH = m_bundleH.slice(0, m_bsize);
        auto bundleA = m_solution.m_alphas.slice(0, m_bsize);
        return nano::remove_if(op, bundleG, bundleH, bundleA);
    }

    void delete_inactive(scalar_t epsilon);
    void delete_smallest(tensor_size_t count);

    void store_aggregate();
    void append_aggregate();
    void append(vector_cmap_t y, vector_cmap_t gy, scalar_t fy, bool serious_step);

    const solution_t& do_solve(scalar_t tau, scalar_t level, const logger_t&);

    // attributes
    quadratic_program_t m_program;  ///< quadratic program
    rsolver_t           m_solver;   ///< solver for the quadratic program
    tensor_size_t       m_bsize{0}; ///< bundle: number of points
    matrix_t            m_bundleG;  ///< bundle: sub-gradients (g_j, -1)_j of shape (size, dims + 1)
    vector_t            m_bundleH;  ///< bundle: function values (f_j + <g_j, x_k^ - x_j>)_j of shape (size,)
    solution_t          m_solution; ///< solution to the quadratic program
    vector_t            m_x;        ///< proximal/stability center (dims)
    vector_t            m_gx;       ///< function gradient at the proximal center (dims)
    scalar_t            m_fx;       ///< function value at the proximal center
    vector_t            m_wlevel;   ///< left-side inequality constraint for the level
};
} // namespace nano
