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
/// \brief utility to select samples for fitting weak learners.
///
class NANO_PUBLIC sampler_t
{
public:
    ///
    /// \brief constructor
    ///
    sampler_t(const indices_t& samples, uint64_t seed);

    ///
    /// \brief returns the samples to use for fitting weak learners.
    ///
    indices_t sample(const tensor2d_t& errors_losses, const tensor4d_t& gradients, const subsample_type);

private:
    // attributes
    const indices_t& m_samples; ///< training samples to select from
    rng_t            m_rng;     ///<
};

///
/// \brief utility to track the optimum boosting round using early stopping on the validation samples.
///
struct NANO_PUBLIC optimum_t
{
    ///
    /// \brief constructor
    ///
    optimum_t(const tensor2d_t& values);

    ///
    /// \brief returns true if early stopping is detected
    ///     (the validation error doesn't decrease significantly in the recent boosting rounds) or
    ///     the training error is too small.
    ///
    bool done(const tensor2d_t& errors_losses, const indices_t& train_samples, const indices_t& valid_samples,
              const rwlearners_t& wlearners, scalar_t epsilon, size_t patience);

    ///
    /// \brief returns the optimum number of boosting rounds.
    ///
    size_t round() const { return m_round; }

    ///
    /// \brief returns the optimum value (the mean error on the validation samples).
    ///
    scalar_t value() const { return m_value; }

    ///
    /// \brief returns the optimum error and loss values for all samples.
    ///
    const tensor2d_t& values() const { return m_values; }

private:
    // attributes
    size_t     m_round{0U};  ///<
    scalar_t   m_value{0.0}; ///<
    tensor2d_t m_values;     ///< optimum (error|loss) for all samples
};
} // namespace nano::gboost
