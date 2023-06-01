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
///     against the ground truth and return the errors and loss values.
///
NANO_PUBLIC tensor2d_t evaluate(const dataset_t&, const indices_t& samples, const loss_t& loss,
                                const tensor2d_t& weights, const tensor1d_t& bias, tensor_size_t batch);

///
/// \brief return the weight magnitude associated to each feature (cumulated over flatten inputs).
/// NB: usually the higher the weights, the more important a feature.
///
NANO_PUBLIC tensor1d_t feature_importance(const dataset_t&, const tensor2d_t& weights);

///
/// \brief return the sparsity ratio of the given feature weight magnitudes
///     computed as the fraction of the feature weights below the given threshold.
///
NANO_PUBLIC scalar_t sparsity_ratio(const tensor1d_t& feature_importance, scalar_t threshold = 1e-6);
} // namespace nano::linear
