#pragma once

#include <nano/solver/status.h>
#include <nano/tensor.h>

namespace nano
{
///
/// \brief models the optimum minimum of a function (the solution of a numerical optimization problem).
///
struct optimum_t
{
    optimum_t() = default;

    explicit optimum_t(vector_t xbest)
        : m_xbest(std::move(xbest))
    {
    }

    auto& fbest(const scalar_t fbest)
    {
        m_fbest = fbest;
        return *this;
    }

    // attributes
    static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

    vector_t      m_xbest;                            ///< optimum: solution
    scalar_t      m_fbest{NaN};                       ///< optimum: criterion
};
} // namespace nano
