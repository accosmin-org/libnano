#pragma once

#include <nano/dataset.h>
#include <nano/loss.h>

namespace nano::linear
{
    ///
    /// \brief compute the predictions of the linear model with the given weights and bias.
    ///
    NANO_PUBLIC void predict(const tensor2d_cmap_t& inputs, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
                             tensor4d_map_t&& outputs);

    ///
    /// \brief compute the predictions of the linear model with the given weights and bias.
    ///
    NANO_PUBLIC void predict(const tensor2d_cmap_t& inputs, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
                             tensor4d_t& outputs);

    ///
    /// \brief evaluate the predictions of the linear model with the given weights and bias
    ///     against the ground truth and returns the errors and loss values.
    ///
    NANO_PUBLIC tensor2d_t evaluate(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                                    const tensor2d_t& weights, const tensor1d_t& bias, tensor_size_t batch);
} // namespace nano::linear
