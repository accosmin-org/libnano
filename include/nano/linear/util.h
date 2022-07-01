#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

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
} // namespace nano::linear
