#pragma once

#include <nano/learner.h>
#include <nano/loss.h>
#include <nano/solver.h>
#include <nano/splitter.h>
#include <nano/tuner.h>

namespace nano
{
    class model_t;
    using rmodel_t = std::unique_ptr<model_t>;

    ///
    /// \brief cross-validation statistics obtained while fitting a ML model.
    ///
    struct fit_result_t
    {
        struct cv_result_t
        {
            cv_result_t();
            cv_result_t(tensor1d_t params, tensor_size_t folds);

            tensor1d_t m_params;                       ///< hyper-parameter values
            tensor1d_t m_train_errors, m_train_values; ///< error and loss values for training samples
            tensor1d_t m_valid_errors, m_valid_values; ///< error and loss values for validation samples
        };

        using cv_results_t = std::vector<cv_result_t>;

        static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

        // attributes
        strings_t    m_param_names;      ///<
        cv_results_t m_cv_results;       ///<
        tensor1d_t   m_refit_params;     ///<
        scalar_t     m_refit_error{NaN}; ///<
        scalar_t     m_refit_value{NaN}; ///<
    };

    ///
    /// \brief interface for machine learning models.
    ///
    /// the minimum set of operations are:
    ///     - training (mutable) which fits the model on the given dataset,
    ///     - prediction (constant) which evaluates the trained model on the given dataset,
    ///     - saving/reading to/from binary streams.
    ///
    class NANO_PUBLIC model_t : public learner_t, public clonable_t<model_t>
    {
    public:
        ///
        /// \brief logging operator: op(fit_result, prefix)
        ///
        using logger_t = std::function<void(const fit_result_t&, const string_t&)>;

        ///
        /// \brief default constructor.
        ///
        explicit model_t(string_t id);

        ///
        /// \brief returns the available implementations.
        ///
        static factory_t<model_t>& all();

        ///
        /// \brief fit the model using the given samples and the current set of (hyper-)parameters
        ///     and returns the average error of the given samples.
        ///
        virtual fit_result_t fit(const dataset_t&, const indices_t&, const loss_t&, const solver_t&, const splitter_t&,
                                 const tuner_t&) = 0;

        ///
        /// \brief evaluate the trained model and returns the predictions for each of the given samples.
        ///
        virtual tensor4d_t predict(const dataset_t&, const indices_t&) const = 0;

        ///
        /// \brief set the logging callback
        ///
        void logger(const logger_t& logger);

    protected:
        void log(const fit_result_t&, const string_t& prefix) const;

    private:
        // attributes
        logger_t m_logger; ///<
    };
} // namespace nano
