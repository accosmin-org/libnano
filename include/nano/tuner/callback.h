#pragma once

#include <functional>
#include <nano/tensor.h>

namespace nano
{
///< evaluates the given set of hyper-parameter values: (trials, hyper-parameter values) => (trials,)
using tuner_callback_t = std::function<tensor1d_t(const tensor2d_t&)>;
} // namespace nano
