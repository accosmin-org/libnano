#include <nano/dataset/iterator.h>
#include <nano/linear/util.h>

using namespace nano;

void linear::predict(const tensor2d_cmap_t& inputs, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
                     tensor4d_map_t&& outputs)
{
    [[maybe_unused]] const auto isize   = weights.cols();
    const auto                  tsize   = weights.rows();
    const auto                  samples = inputs.size<0>();

    assert(tsize == bias.size());
    assert(samples == inputs.size<0>());
    assert(samples == outputs.size<0>());
    assert(samples * isize == inputs.size());
    assert(samples * tsize == outputs.size());

    outputs.reshape(samples, tsize).matrix() = inputs.matrix() * weights.matrix().transpose();
    outputs.reshape(samples, tsize).matrix().rowwise() += bias.vector().transpose();
}

void linear::predict(const tensor2d_cmap_t& inputs, const tensor2d_cmap_t& weights, const tensor1d_cmap_t& bias,
                     tensor4d_t& outputs)
{
    outputs.resize(inputs.size<0>(), bias.size(), 1, 1);
    predict(inputs, weights, bias, outputs.tensor());
}

tensor2d_t linear::evaluate(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                            const tensor2d_t& weights, const tensor1d_t& bias, const tensor_size_t batch,
                            const size_t threads)
{
    auto iterator = flatten_iterator_t{dataset, samples, threads};
    iterator.scaling(scaling_type::none);
    iterator.batch(batch);

    tensor2d_t values(2, samples.size());
    tensor4d_t outputs(cat_dims(samples.size(), dataset.target_dims()));
    iterator.loop(
        [&](tensor_range_t range, size_t, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
        {
            ::nano::linear::predict(inputs, weights, bias, outputs.slice(range));
            loss.error(targets, outputs.slice(range), values.tensor(0).slice(range));
            loss.value(targets, outputs.slice(range), values.tensor(1).slice(range));
        });

    return values;
}
