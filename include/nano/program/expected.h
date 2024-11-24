#pragma once

#include <nano/solver/status.h>
#include <nano/tensor.h>

namespace nano::program
{
///
/// \brief models the expected solution of a linear or quadratic program.
///
struct expected_t
{
    expected_t() = default;

    explicit expected_t(vector_t xbest)
        : m_xbest(std::move(xbest))
    {
    }

    auto& x0(vector_t x0)
    {
        m_x0 = std::move(x0);
        return *this;
    }

    auto& status(const solver_status status)
    {
        m_status = status;
        return *this;
    }

    auto& epsilon(const scalar_t epsilon)
    {
        m_epsilon = epsilon;
        return *this;
    }

    auto& fbest(const scalar_t fbest)
    {
        m_fbest = fbest;
        return *this;
    }

    // attributes
    static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

    vector_t      m_xbest;                            ///< optimum: solution
    vector_t      m_vbest;                            ///< optimum: lagrange multiplier for the equality constraints
    scalar_t      m_fbest{NaN};                       ///< optimum: criterion
    vector_t      m_x0;                               ///< optional starting point
    scalar_t      m_epsilon{1e-8};                    ///< precision to
    solver_status m_status{solver_status::converged}; ///< expected solver status
};
} // namespace nano::program
