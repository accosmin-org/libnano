#pragma once

#include <functional>
#include <nano/tensor.h>

namespace nano
{
///
/// \brief callback to evaluate the given set of hyper-parameter values:
///     in:  hyper-parameter values of shape (trials, number of hyper-parameters) =>
///     out: goodness for each trial of shape (trials,)
///
using tuner_callback_t = std::function<tensor1d_t(const tensor2d_t&)>;
} // namespace nano
