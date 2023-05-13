#pragma once

#include <nano/solver/state.h>
#include <nano/tensor.h>

namespace nano::linear
{
///
/// \brief linear model and optimization statistics.
///
struct NANO_PUBLIC fit_result_t
{
    enum class stats
    {
        solver_fcalls,  ///< number of function value calls by the solver
        solver_gcalls,  ///< number of function gradient calls by the solver
        solver_status_, ///< solver_status enumeration produced by the solver
    };

    ///
    /// \brief default constructor
    ///
    fit_result_t();

    ///
    /// \brief constructor
    ///
    explicit fit_result_t(tensor1d_t bias, tensor2d_t weights, const solver_state_t&);

    // attributes
    tensor1d_t m_bias;       ///<
    tensor2d_t m_weights;    ///<
    tensor1d_t m_statistics; ///< (statistics indexed by the associated enumeration)
};
} // namespace nano::linear
