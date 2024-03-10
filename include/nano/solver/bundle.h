#pragma once

#include <nano/program/solver.h>
#include <nano/solver/state.h>
#include <nano/tensor/algorithm.h>

namespace nano
{
///
/// \brief models the bundle of sub-gradients as used by penalized (proximal) bundle algorithms.
///
/// see (1) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
///
/// NB: the bundle is kept small by removing all inactive constraints and the oldest ones if needed.
///
class NANO_PUBLIC bundle_t
{
public:
    ///
    /// \brief constructor
    ///
    bundle_t(const solver_state_t&, tensor_size_t max_size);

    ///
    /// \brief
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief
    ///
    static bundle_t make(const solver_state_t&, const configurable_t&, const string_t& prefix);

    tensor_size_t dims() const { return m_bundleS.size<1>(); }

    auto size() const { return m_size; }

    auto capacity() const { return m_alphas.size(); }

    const auto& x() const { return m_x; }

    const auto& gx() const { return m_gx; }

    auto fx() const { return m_fx; }

    auto S() const { return m_bundleS.slice(0, size()); }

    auto e() const
    {
        auto e = m_bundleE.slice(0, size());
        assert(e.min() + epsilon1<scalar_t>() > 0.0);
        return e;
    }

    auto alpha() const { return m_alphas.slice(0, size()); }

    auto smeared_e() const { return e().dot(alpha()); }

    auto smeared_s() const { return S().transpose() * alpha(); }

    auto delta(const scalar_t miu) const
    {
        const auto delta = smeared_e() + 1.0 / (2.0 * miu) * smeared_s().squaredNorm();
        assert(delta + epsilon1<scalar_t>() >= 0.0);
        return delta;
    }

    auto proximal(const scalar_t miu) const { return m_x - smeared_s() / miu; }

    ///
    /// \brief change the proximity center to the given point and update the bundle.
    ///
    void moveto(const vector_cmap_t y, const vector_cmap_t gy, scalar_t fy);

    ///
    /// \brief update the bundle with the given point.
    ///
    void append(const vector_cmap_t y, const vector_cmap_t gy, scalar_t fy);

    ///
    /// \brief return the solution of the penalized proximal bundle problem.
    ///
    void solve(scalar_t miu);

private:
    template <typename toperator>
    tensor_size_t remove_if(const toperator& op)
    {
        return nano::remove_if(op, m_bundleE.slice(0, m_size), m_bundleS.slice(0, m_size), m_alphas.slice(0, m_size));
    }

    void delete_oldest(tensor_size_t count = 2);
    void delete_smallest(tensor_size_t count = 2);
    void delete_inactive(scalar_t epsilon = epsilon1<scalar_t>());

    void store_aggregate();
    void append_aggregate();
    void append(const vector_cmap_t y, const vector_cmap_t gy, scalar_t fy, bool serious_step);

    // attributes
    program::solver_t m_solver;  ///< buffer: quadratic program solver
    tensor_size_t     m_size{0}; ///< bundle: number of points
    matrix_t          m_bundleS; ///< bundle: sub-gradients (size, dims)
    vector_t          m_bundleE; ///< bundle: linearized errors (size)
    vector_t          m_alphas;  ///< optimal Lagrange multipliers (size)
    vector_t          m_x;       ///< proximal center (dims)
    vector_t          m_gx;      ///< proximal center's gradient (dims)
    scalar_t          m_fx;      ///< proximal center's function value
};
} // namespace nano
