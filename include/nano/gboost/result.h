#pragma once

#include <nano/solver/state.h>
#include <nano/wlearner.h>

namespace nano::gboost
{
    ///
    /// \brief boosting model and per boosting round statistics with support for early stopping.
    ///
    struct NANO_PUBLIC fit_result_t
    {
        enum class stats
        {
            train_error = 0,
            train_loss,
            valid_error,
            valid_loss,
            solver_fcalls,
            solver_gcalls,
            solver_status_,
        };

        ///
        /// \brief constructor
        ///
        fit_result_t(tensor_size_t max_rounds = 0);

        ///
        /// \brief enable moving.
        ///
        fit_result_t(fit_result_t&&);
        fit_result_t(const fit_result_t&);

        ///
        /// \brief enable copying.
        ///
        fit_result_t& operator=(const fit_result_t&);
        fit_result_t& operator=(fit_result_t&&);

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
        /// \brief trim the resuupdate statistics for the given boosting round.
        ///
        void done(tensor_size_t optimum_round);

        // attributes
        tensor1d_t   m_bias;
        rwlearners_t m_wlearners;
        tensor2d_t   m_statistics; ///< (boosting round, statistics indexed by the associated enumeration)
    };
} // namespace nano::gboost
