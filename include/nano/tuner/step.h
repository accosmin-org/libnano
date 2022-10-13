#pragma once

#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief models a step (trial) computed when tuning hyper-parameters of a machine learning model.
    ///
    struct NANO_PUBLIC tuner_step_t
    {
        tuner_step_t();
        tuner_step_t(indices_t igrid, tensor1d_t param, scalar_t value);

        static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

        indices_t  m_igrid;      ///< grid indices of the hyper-parameter values
        tensor1d_t m_param;      ///< hyper-parameter values (mapping of indices to the grid)
        scalar_t   m_value{NaN}; ///< associated evaluation score (the lower the better)
    };

    using tuner_steps_t = std::vector<tuner_step_t>;

    inline bool operator<(const tuner_step_t& lhs, const tuner_step_t& rhs)
    {
        return lhs.m_value < rhs.m_value;
    }
} // namespace nano
