#pragma once

#include <nano/solver/state.h>
#include <nano/wlearner.h>

namespace nano::gboost
{
///
/// \brief results collected by fitting a gradient boosting model for a given set of hyper-parameter values
///     and a training-validation split.
///
struct NANO_PUBLIC result_t
{
    enum class stats : uint8_t
    {
        train_error,    ///< mean training error
        train_loss,     ///< mean training loss value
        valid_error,    ///< mean validation error
        valid_loss,     ///< mean validation loss value
        shrinkage,      ///< selected shrinkage ratio (to regularize the validation loss)
        solver_fcalls,  ///< number of function value calls by the solver
        solver_gcalls,  ///< number of function gradient calls by the solver
        solver_status_, ///< solver_status enumeration produced by the solver
    };

    ///
    /// \brief constructor
    ///
    explicit result_t(const tensor2d_t* errors_values = nullptr, const indices_t* train_samples = nullptr,
                      const indices_t* valid_samples = nullptr, tensor_size_t max_rounds = 0);

    ///
    /// \brief enable moving.
    ///
    result_t(result_t&&) noexcept;
    result_t(const result_t&);

    ///
    /// \brief enable copying.
    ///
    result_t& operator=(const result_t&);
    result_t& operator=(result_t&&) noexcept;

    ///
    /// \brief destructor
    ///
    ~result_t();

    ///
    /// \brief update statistics for the given boosting round.
    ///
    void update(tensor_size_t round, scalar_t shrinkage_ratio, const solver_state_t&);
    void update(tensor_size_t round, scalar_t shrinkage_ratio, const solver_state_t&, rwlearner_t&&);

    ///
    /// \brief trim the statistics at the given boosting round (selected by early stopping).
    ///
    void done(tensor_size_t optimum_round);

    // attributes
    const tensor2d_t* m_errors_values{nullptr}; ///< (error|loss, sample) evaluation results
    const indices_t*  m_train_samples{nullptr}; ///< training samples
    const indices_t*  m_valid_samples{nullptr}; ///< validation samples
    tensor1d_t        m_bias;                   ///< bias prediction
    rwlearners_t      m_wlearners;              ///< selected weak learners
    tensor2d_t        m_statistics;             ///< (boosting round, statistics indexed by the associated enumeration)
};
} // namespace nano::gboost
