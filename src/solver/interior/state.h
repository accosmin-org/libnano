#pragma once

#include <nano/tensor.h>

namespace nano
{
struct state_t
{
    state_t(vector_t x0, vector_t u0, vector_t v0)
        : m_x(std::move(x0))
        , m_u(std::move(u0))
        , m_v(std::move(v0))
        , m_dx(vector_t::constant(m_x.size(), 0.0))
        , m_du(vector_t::constant(m_u.size(), 0.0))
        , m_dv(vector_t::constant(m_v.size(), 0.0))
        , m_rdual(vector_t::constant(m_x.size(), 0.0))
        , m_rcent(vector_t::constant(m_u.size(), 0.0))
        , m_rprim(vector_t::constant(m_v.size(), 0.0))
    {
    }

    bool valid() const
    {
        return m_dx.all_finite() && m_dv.all_finite() && m_du.all_finite() && std::isfinite(m_eta) &&
               m_rdual.all_finite() && m_rcent.all_finite() && m_rprim.all_finite();
    }

    scalar_t residual() const
    {
        return m_rdual.lpNorm<Eigen::Infinity>() + ///<
               m_rcent.lpNorm<Eigen::Infinity>() + ///<
               m_rprim.lpNorm<Eigen::Infinity>();  ///<
    }

    // attributes
    vector_t m_x;        ///< solution (primal problem)
    vector_t m_u;        ///< Lagrange multipliers (inequality constraints)
    vector_t m_v;        ///< Lagrange multipliers (equality constraints)
    vector_t m_dx;       ///<
    vector_t m_du;       ///<
    vector_t m_dv;       ///<
    scalar_t m_eta{0.0}; ///< surrogate duality gap
    vector_t m_rdual;    ///< dual residual
    vector_t m_rcent;    ///< central residual
    vector_t m_rprim;    ///< primal residual
};
} // namespace nano
