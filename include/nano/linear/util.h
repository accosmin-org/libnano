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

    ///
    /// \brief iterate through all samples of a fold using the given batch size and call the given operator
    ///     with the [begin, end) range of samples like: toperator(inputs, targets, outputs, begin, end, tnum).
    ///
    template <typename toperator>
    void iterate(const iterator_t& iterator, const fold_t& fold, const tensor_size_t batch,
        const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias, const toperator& op)
    {
        const auto samples = iterator.samples(fold);

        assert(bias.size() == weights.cols());
        assert(::nano::size(iterator.idim()) == weights.rows());
        assert(::nano::size(iterator.tdim()) == weights.cols());

        std::vector<tensor4d_t> inputs{tpool_t::size()};
        std::vector<tensor4d_t> outputs{tpool_t::size()};
        std::vector<tensor4d_t> targets{tpool_t::size()};

        loopr(samples, batch, [&] (const tensor_size_t begin, const tensor_size_t end, const tensor_size_t tnumi)
        {
            const auto tnum = static_cast<size_t>(tnumi);

            iterator.inputs(fold, begin, end, inputs[tnum]);
            assert(inputs[tnum].size<0>() == end - begin);
            assert(inputs[tnum].size() == (end - begin) * weights.rows());

            iterator.targets(fold, begin, end, targets[tnum]);
            assert(targets[tnum].size<0>() == end - begin);
            assert(targets[tnum].size() == (end - begin) * weights.cols());

            predict(inputs[tnum], weights, bias, outputs[tnum]);

            op(inputs[tnum], targets[tnum], outputs[tnum], begin, end, tnum);
        });
    }
}}
