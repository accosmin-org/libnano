#pragma once

#include <nano/program/solver.h>
#include <nano/solver/state.h>
#include <nano/tensor/algorithm.h>

namespace nano
{
///
/// \brief bundle pruning strategies.
///
enum class bundle_pruning
{
    oldest_point, ///< see (1) - the oldest point in the bundle
    largest_error ///< see (2) - the point with the largest approximation error
};

///
/// \brief models the bundle of sub-gradients as used by penalized (proximal) bundle algorithms.
///
/// see (1) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
/// see (2) "Variable metric bundle methods: from conceptual to implementable forms", by Lemarechal, Sagastizabal, 1997
///
/// NB: the bundle is kept small by removing all inactive constraints and the oldest ones if needed.
///
class NANO_PUBLIC bundle_t
{
public:
    ///
    /// \brief constructor
    ///
    bundle_t(const solver_state_t&, tensor_size_t max_size, bundle_pruning);

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
    /// \brief return the smeared approximation error, see (1).
    ///
    auto smeared_e() const { return e().dot(alpha()); }

    ///
    /// \brief return the smeared sub-gradient, see (1).
    ///
    auto smeared_s() const { return S().transpose() * alpha(); }

    ///
    /// \brief return the approximation error, see (1).
    ///
    scalar_t delta(const matrix_t& invM, scalar_t miu) const;

    ///
    /// \brief return the estimated proximal point.
    ///
    auto proximal(const matrix_t& invM, const scalar_t miu) const
    {
        assert(std::isfinite(miu));
        assert(miu > 0.0);
        return m_x - (invM * smeared_s()) / miu;
    }

    ///
    /// \brief change the proximity center to the given point and update the bundle.
    ///
    void moveto(vector_cmap_t y, vector_cmap_t gy, scalar_t fy);

    ///
    /// \brief update the bundle with the given point.
    ///
    void append(vector_cmap_t y, vector_cmap_t gy, scalar_t fy);

    ///
    /// \brief return the solution of the penalized proximal bundle problem, see (1).
    ///
    void solve(const matrix_t& invM, scalar_t miu);

private:
    tensor_size_t dims() const { return m_bundleS.size<1>(); }

    tensor_size_t capacity() const { return m_alphas.size(); }

    vector_cmap_t alpha() const { return m_alphas.slice(0, size()); }

    matrix_cmap_t S() const { return m_bundleS.slice(0, size()); }

    vector_cmap_t e() const
    {
        auto e = m_bundleE.slice(0, size());
        assert(e.min() + epsilon1<scalar_t>() > 0.0);
        return e;
    }

    template <typename toperator>
    tensor_size_t remove_if(const toperator& op)
    {
        return nano::remove_if(op, m_bundleE.slice(0, m_size), m_bundleS.slice(0, m_size), m_alphas.slice(0, m_size));
    }

    void delete_oldest(tensor_size_t count = 2);
    void delete_largest(tensor_size_t count = 2);
    void delete_inactive(scalar_t epsilon = epsilon1<scalar_t>());

    void store_aggregate();
    void append_aggregate();
    void append(vector_cmap_t y, vector_cmap_t gy, scalar_t fy, bool serious_step);

    // attributes
    program::solver_t m_solver;                                ///< buffer: quadratic program solver
    tensor_size_t     m_size{0};                               ///< bundle: number of points
    matrix_t          m_bundleS;                               ///< bundle: sub-gradients (size, dims)
    vector_t          m_bundleE;                               ///< bundle: linearized errors (size)
    vector_t          m_alphas;                                ///< optimal Lagrange multipliers (size)
    vector_t          m_x;                                     ///< proximal center (dims)
    vector_t          m_gx;                                    ///< proximal center's gradient (dims)
    scalar_t          m_fx;                                    ///< proximal center's function value
    bundle_pruning    m_pruning{bundle_pruning::oldest_point}; ///<
};
} // namespace nano
