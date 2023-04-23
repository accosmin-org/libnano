#pragma once

#include <nano/dataset/iterator.h>
#include <nano/loss.h>
#include <nano/wlearner.h>

namespace nano::gboost
{
///
/// \brief evaluate the predictions (at a given boosting round) against the targets.
///
NANO_PUBLIC void evaluate(const targets_iterator_t& iterator, const loss_t& loss, const tensor4d_t& outputs,
                          tensor2d_t& values);

///
/// \brief returns the mean loss value for the given samples.
///
NANO_PUBLIC scalar_t mean_loss(const tensor2d_t& errors_losses, const indices_t& samples);

///
/// \brief returns the mean error value for the given samples.
///
NANO_PUBLIC scalar_t mean_error(const tensor2d_t& errors_losses, const indices_t& samples);

///
/// \brief returns true if early stopping is detected
///     (the validation error doesn't decrease significantly in the recent boosting rounds) or
///     the training error is too small.
///
NANO_PUBLIC bool done(const tensor2d_t& errors_losses, const indices_t& train_samples, const indices_t& valid_samples,
                      const rwlearners_t& wlearners, scalar_t epsilon, size_t patience, size_t& optimum_round,
                      scalar_t& optimum_value, tensor2d_t& optimum_values);
} // namespace nano::gboost
