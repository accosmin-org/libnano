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
