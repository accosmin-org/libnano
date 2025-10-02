#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

namespace nano::linear
{
///
/// \brief cumulates partial results per thread useful in evaluating the linear functions.
///
class NANO_PUBLIC accumulator_t
{
public:
    ///
    /// \brief default constructor
    ///
    accumulator_t();

    ///
    /// \brief constructor
    ///
    accumulator_t(tensor_size_t isize, tensor_size_t tsize);

    ///
    /// \brief reset the accumulator.
    ///
    void clear();

    ///
    /// \brief cumulate partial results.
    ///
    accumulator_t& operator+=(const accumulator_t& other);

    ///
    /// \brief normalize the cumulated results with the given number of samples.
    ///
    accumulator_t& operator/=(tensor_size_t samples);

    // attributes
    tensor4d_t m_outputs; ///< predictions
    tensor1d_t m_loss_fx; ///< loss values
    tensor4d_t m_loss_gx; ///< loss gradients wrt outputs
    tensor7d_t m_loss_hx; ///< loss hessians wrt outputs
    scalar_t   m_fx{0};   ///< sum of loss values
    tensor1d_t m_gb;      ///< sum of loss gradients wrt bias
    tensor2d_t m_gw;      ///< sum of loss gradients wrt weights
    tensor2d_t m_hww;     ///< sum of loss hessians wrt weigths+bias
    tensor2d_t m_hwb;     ///< sum of loss hessians wrt weigths+bias
    tensor2d_t m_hbb;     ///< sum of loss hessians wrt weigths+bias
};

using accumulators_t = std::vector<accumulator_t>;
} // namespace nano::linear
