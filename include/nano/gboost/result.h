#pragma once

#include <nano/solver/state.h>
#include <nano/wlearner.h>

namespace nano::gboost
{
///
/// \brief boosting model and per boosting round statistics.
///
struct NANO_PUBLIC fit_result_t
{
    enum class stats
    {
        train_error = 0, ///< mean training error
        train_loss,      ///< mean training loss value
        valid_error,     ///< mean validation error
        valid_loss,      ///< mean validation loss value
        solver_fcalls,   ///< number of function value calls by the solver
        solver_gcalls,   ///< number of function gradient calls by the solver
        solver_status_,  ///< solver_status enumeration produced by the solver
    };

    ///
    /// \brief constructor
    ///
    explicit fit_result_t(tensor_size_t max_rounds = 0);

    ///
    /// \brief enable moving.
    ///
    fit_result_t(fit_result_t&&) noexcept;
    fit_result_t(const fit_result_t&);

    ///
    /// \brief enable copying.
    ///
    fit_result_t& operator=(const fit_result_t&);
    fit_result_t& operator=(fit_result_t&&) noexcept;

    ///
    /// \brief destructor
    ///
    ~fit_result_t();

    ///
    /// \brief update statistics for the given boosting round.
    ///
    void update(tensor_size_t round, const tensor2d_t& errors_values, const indices_t& train_samples,
                const indices_t& valid_samples, const solver_state_t&);

    ///
    /// \brief update statistics for the given boosting round.
    ///
    void update(tensor_size_t round, const tensor2d_t& errors_values, const indices_t& train_samples,
                const indices_t& valid_samples, const solver_state_t&, rwlearner_t&&);

    ///
    /// \brief trim the statistics at the given boosting round (selected by early stopping).
    ///
    void done(tensor_size_t optimum_round);

    // attributes
    tensor1d_t   m_bias;
    rwlearners_t m_wlearners;
    tensor2d_t   m_statistics; ///< (boosting round, statistics indexed by the associated enumeration)
};
} // namespace nano::gboost
