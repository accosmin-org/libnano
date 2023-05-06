#pragma once

#include <nano/dataset/iterator.h>
#include <nano/loss.h>
#include <nano/wlearner.h>

namespace nano::gboost
{
///
/// \brief evaluate the predictions (at a given boosting round) against the targets.
///
NANO_PUBLIC void evaluate(const targets_iterator_t&, const loss_t&, const tensor4d_t& outputs, tensor2d_t& values);

///
/// \brief tune the shrinkage ratio to optimize the predictions on the given (validation) samples.
///
NANO_PUBLIC scalar_t tune_shrinkage(const targets_iterator_t&, const loss_t&, const tensor4d_t& outputs,
                                    const tensor4d_t& woutputs);

///
/// \brief returns the mean loss value for the given samples.
///
NANO_PUBLIC scalar_t mean_loss(const tensor2d_t& errors_losses, const indices_t& samples);

///
/// \brief returns the mean error value for the given samples.
///
NANO_PUBLIC scalar_t mean_error(const tensor2d_t& errors_losses, const indices_t& samples);
} // namespace nano::gboost
