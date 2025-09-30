#pragma once

#include <nano/tensor.h>

namespace nano::gboost
{
///
/// \brief cumulates partial results per thread useful in evaluating gradient boosting-related functions.
///
class NANO_PUBLIC accumulator_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit accumulator_t(tensor_size_t tsize = 1);

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

    ///
    /// \brief return the cumulated loss values and optionally the cumulated gradients and hessians.
    ///
    scalar_t value(vector_map_t gx, matrix_map_t hx) const;

    // attributes
    tensor4d_t m_outputs; ///< predictions
    tensor1d_t m_loss_fx; ///< loss values
    tensor4d_t m_loss_gx; ///< loss gradients wrt outputs
    tensor7d_t m_loss_hx; ///< loss hessians wrt outputs
    scalar_t   m_fx{0};   ///< sum of loss values
    vector_t   m_gx;      ///< sum of loss gradients wrt outputs
    matrix_t   m_hx;      ///< sum of loss hessians wrt outputs
};

using accumulators_t = std::vector<accumulator_t>;
} // namespace nano::gboost
