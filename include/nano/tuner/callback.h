#pragma once

#include <functional>
#include <nano/tensor.h>

namespace nano
{
///
/// \brief callback to evaluate the given set of hyper-parameter values:
///     hyper-parameter values of shape (trials, number of hyper-parameters) =>
///     goodness of shaoe (trials,)
///
using tuner_callback_t = std::function<tensor1d_t(const tensor2d_t&)>;
} // namespace nano
