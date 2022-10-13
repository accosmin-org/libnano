#pragma once

#include <functional>
#include <nano/tensor.h>

namespace nano
{
    ///< evaluates the candidate hyper-parameter values
    using tuner_callback_t = std::function<tensor1d_t(const tensor2d_t&)>;
} // namespace nano
