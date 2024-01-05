#pragma once

#include <nano/solver/status.h>
#include <nano/tensor.h>

namespace nano::program
{
///
/// \brief the state of a primal-dual interior-point solver.
///
struct NANO_PUBLIC solver_state_t
{
    ///
    /// \brief default constructor
    ///
    solver_state_t() = default;

    ///
    /// \brief constructor
    ///
    solver_state_t(const tensor_size_t n, const tensor_size_t n_ineqs, const tensor_size_t n_eqs)
        : m_x(vector_t::constant(n, nan))
        , m_u(vector_t::constant(n_ineqs, nan))
        , m_v(vector_t::constant(n_eqs, nan))
        , m_rdual(vector_t::constant(n, nan))
        , m_rcent(vector_t::constant(n_ineqs, nan))
        , m_rprim(vector_t::constant(n_eqs, nan))
    {
    }

    ///
    /// \brief return the cumulated residual.
    ///
    scalar_t residual() const
    {
        const auto edual = m_rdual.lpNorm<Eigen::Infinity>();
        const auto ecent = m_rcent.lpNorm<Eigen::Infinity>();
        const auto eprim = m_rprim.lpNorm<Eigen::Infinity>();
        return std::max({edual, ecent, eprim});
    }

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
    solver_status m_status{solver_status::max_iters}; ///< optimization status
    scalar_t      m_ldlt_rcond{0};                    ///< LDLT decomp: reciprocal condition number
    bool          m_ldlt_positive{false};             ///< LDLT decomp: positive semidefinite?, otherwise unstable
};
} // namespace nano::program
