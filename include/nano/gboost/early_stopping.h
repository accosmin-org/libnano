#pragma once

#include <nano/wlearner.h>

namespace nano::gboost
{
///
/// \brief utility to track the optimum boosting round using early stopping on the validation samples.
///
class NANO_PUBLIC early_stopping_t
{
public:
    ///
    /// \brief constructor
    ///
    early_stopping_t(const tensor2d_t& values);

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
