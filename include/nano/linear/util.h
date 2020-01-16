#pragma once

#include <nano/loss.h>
#include <nano/tpool.h>
#include <nano/iterator.h>

namespace nano { namespace linear
{
    ///
    /// \brief compute the predictions of the linear model with the given weights and bias.
    ///
    inline void predict(const tensor4d_cmap_t& inputs, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
        tensor4d_map_t&& outputs)
    {
        const auto isize = weights.rows();
        const auto tsize = weights.cols();
        const auto samples = inputs.size<0>();

        assert(tsize == bias.size());
        assert(samples == inputs.size<0>());
        assert(samples == outputs.size<0>());
        assert(samples * isize == inputs.size());
        assert(samples * tsize == outputs.size());

        outputs.reshape(samples, tsize).matrix() = inputs.reshape(samples, isize).matrix() * weights.matrix();
        outputs.reshape(samples, tsize).matrix().rowwise() += bias.vector().transpose();
    }

    ///
    /// \brief compute the predictions of the linear model with the given weights and bias.
    ///
    inline void predict(const tensor4d_cmap_t& inputs, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
        tensor4d_t& outputs)
    {
        outputs.resize(inputs.size<0>(), bias.size(), 1, 1);
        predict(inputs, weights, bias, outputs.tensor());
    }
}}
