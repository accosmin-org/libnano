#pragma once

#include <nano/logger.h>
#include <nano/program/solver.h>
#include <nano/solver/state.h>
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
///
/// NB: the implementation follows the notation from (1). if the level parameter is infinite,
///     then this formulation becames the penalized proximal bundle algorithms
///     (see 2, RQB from 3, mRQB from 4, FPBA1/FPBA2 from 5).
///
/// NB: the bundle is kept small by:
///     - first removing all inactive constraints and
///     - then the ones with the largest approximation error if needed - see (2).
///
/// FIXME: implement removing oldest constraints as well.
///
class NANO_PUBLIC bundle_t
{
public:
    struct solution_t
    {
        vector_t m_x;      ///< optimum: stability center
        scalar_t m_r;      ///< optimum: level
        scalar_t m_lambda; ///< lagrangian multiplier associated to the condition r <= l_k (level)
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
    tensor_size_t size() const { return m_size; }

    ///
    /// \brief return the proximity center.
    ///
    const vector_t& x() const { return m_x; }

    ///
    /// \brief return the sub-gradient at the proximity center.
    ///
    const vector_t& gx() const { return m_gx; }

    ///
    /// \brief return the function value at the proximity center.
    ///
    scalar_t fx() const { return m_fx; }

    ///
    /// \brief return the approximation error, see (1).
    ///
    scalar_t delta(const scalar_t miu) const
    {
        const auto delta = smeared_e() + 1.0 / (2.0 * miu) * smeared_s().squaredNorm();
        assert(delta + epsilon1<scalar_t>() >= 0.0);
        return delta;
    }

    ///
    /// \brief return the estimated proximal point.
    ///
    auto proximal() const { return m_x - smeared_s() / miu; }

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
    ///     argmin_(x, r) r + ||x - x_k||^2 / (2 * tau)
    ///             s.t.  f(x_j) + <g_j, x - x_j> <= r (for all sub-gradients j in the bundle)
    ///             s.t.  r <= l_k (the level parameter).
    ///
    ///     where x_k is the current proximal stability center.
    ///
    void solve(scalar_t tau, scalar_t level, const logger_t&);

    ///
    /// \brief return the aggregate linearization error, see (1).
    ///
    scalar_t aggregate_error() const;

    ///
    /// \brief return the aggregate sub-gradient, see (1).
    ///
    auto aggregate_gradient() const;

    scalar_t predicted_descent() const; // v_k^tau

    ///
    /// \brief return true if converged wrt the smeared approximation error, see (1).
    /// FIXME: citation here, but at least use it consistently across proximal bundle algorithms.
    ///
    bool econverged(scalar_t epsilon) const;

    ///
    /// \brief return true if converged wrt the smeared sub-gradient, see (1).
    /// FIXME: citation here, but at least use it consistently across proximal bundle algorithms.
    ///
    bool sconverged(scalar_t epsilon) const;

private:
    tensor_size_t dims() const { return m_x.size(); }

    tensor_size_t capacity() const { return m_bundlef.size(); }

    matrix_cmap_t bundleG() const { return m_bundleG.slice(0, m_size); }

    vector_cmap_t bundlef() const { return m_bundlef.slice(0, m_size); }

    template <class toperator>
    tensor_size_t remove_if(const toperator& op)
    {
        return nano::remove_if(op, bundleG(), bundlef());
    }

    void delete_inactive(scalar_t epsilon);
    void delete_largest(tensor_size_t count);
    void delete_oldest(tensor_size_t count);

    void store_aggregate();
    void append_aggregate();
    void append(vector_cmap_t y, vector_cmap_t gy, scalar_t fy, bool serious_step);

    // attributes
    using quadratic_program_t = program::quadratic_program_t;

    quadratic_program_t m_program; ///< buffer: quadratic program definition
    program::solver_t   m_solver;  ///< buffer: quadratic program solver
    tensor_size_t       m_size{0}; ///< bundle: number of points
    matrix_t            m_bundleG; ///< bundle: sub-gradients (g_j, -1)_j of shape (size, dims + 1)
    vector_t            m_bundlef; ///< bundle: function values (-f_j)_j of shape (size,)
    vector_t            m_optixr;  ///< solution to the quadratic bundle problem: (x, r) of shape (dims + 1,)
    vector_t            m_x;       ///< proximal/stability center (dims)
    vector_t            m_gx;      ///< function gradient at the proximal center (dims)
    scalar_t            m_fx;      ///< function value at the proximal center
};
} // namespace nano
