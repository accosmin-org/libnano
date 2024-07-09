#pragma once

#include <nano/mlearn/params.h>
#include <nano/mlearn/result.h>

namespace nano::ml
{
///
/// \brief callback to evaluate the given set of hyper-parameter values:
///     (training samples, validation samples, hyper-parameter values, previous relevant model) =>
///     (errors and loss function values for training samples, same for validation samples, model)
///
using tune_callback_t = std::function<std::tuple<tensor2d_t, tensor2d_t, std::any>(const indices_t&, const indices_t&,
                                                                                   tensor1d_cmap_t, std::any)>;

///
/// \brief tune hyper-parameters required to fit a machine learning model.
///
/// NB: each set of hyper-parameter values is evaluated using the given callback.
/// NB: the tuning is performed in parallel across the current set of hyper-parameter values to evaluate and the folds.
///
NANO_PUBLIC result_t tune(const string_t& prefix, const indices_t& samples, const params_t&, param_spaces_t,
                          const tune_callback_t&);
} // namespace nano::ml
