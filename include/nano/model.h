#pragma once

#include <nano/loss.h>
#include <nano/solver.h>
#include <nano/generator.h>
#include <nano/core/serializable.h>

namespace nano
{
    class model_t;
    using model_factory_t = factory_t<model_t>;
    using rmodel_t = model_factory_t::trobject;
    using rmodels_t = std::vector<rmodel_t>;

    ///
    /// \brief stores values for a set of parameters given by name, optionally with:
    ///     - the validation error (e.g. from k-fold cross-validation).
    ///
    class NANO_PUBLIC model_config_t
    {
    public:

        using xvalue_t = std::pair<string_t, std::variant<int64_t, scalar_t>>;
        using xvalues_t = std::vector<xvalue_t>;

        ///
        /// \brief default constructor
        ///
        model_config_t() = default;

        ///
        /// \brief store parameter's value.
        ///
        void add(string_t name, int32_t value);
        void add(string_t name, int64_t value);
        void add(string_t name, scalar_t value);

        ///
        /// \brief store validation error.
        ///
        void evaluate(scalar_t error);

        ///
        /// \brief access functions.
        ///
        auto error() const { return m_error; }
        const auto& values() const { return m_values; }

    private:

        // attributes
        xvalues_t       m_values;           ///<
        scalar_t        m_error{std::numeric_limits<scalar_t>::quiet_NaN()};    ///<
    };

    using model_configs_t = std::vector<model_config_t>;

    NANO_PUBLIC std::ostream& operator<<(std::ostream&, const model_config_t&);

    ///
    /// \brief interface for machine learning models.
    ///
    /// the minimum set of operations are:
    ///     - training (mutable) which fits the model on the given dataset,
    ///     - prediction (constant) which evaluates the trained model on the given dataset,
    ///     - saving/reading to/from binary streams.
    ///
    class NANO_PUBLIC model_t : public serializable_t
    {
    public:

        ///
        /// \brief returns the available implementations.
        ///
        static model_factory_t& all();

        ///
        /// \brief default constructor.
        ///
        model_t();

        ///
        /// \brief @see serializable_t
        ///
        void read(std::istream&) override;

        ///
        /// \brief @see serializable_t
        ///
        void write(std::ostream&) const override;

        ///
        /// \brief clone the object.
        ///
        virtual rmodel_t clone() const = 0;

        ///
        /// \brief fit the model using the given samples and the current set of (hyper-)parameters
        ///     and returns the average error of the given samples.
        ///
        scalar_t fit(const dataset_generator_t&, const indices_t&, const loss_t&, const solver_t&) = 0;

        ///
        /// \brief evaluate the trained model and returns the error for each of the given samples.
        ///
        tensor1d_t evaluate(const dataset_generator_t&, const indices_t&, const loss_t&) const;

        ///
        /// \brief evaluate the trained model and returns the predictions for each of the given samples.
        ///
        tensor4d_t predict(const dataset_generator_t&, const indices_t&) const;

        ///
        /// \brief returns all stored parameters.
        ///
        const auto& params() const { return m_params; }
        model_config_t config() const;

    private:

        void compatible(const dataset_generator_t&) const;

        indices_t do_fit(const dataset_generator_t&, const indices_t&, const loss_t&, const solver_t&) = 0;
        tensor4d_t do_predict(const dataset_generator_t&, const indices_t&) const = 0;

        // attributes
        features_t      m_inputs;       ///< input features
        feature_t       m_target;       ///< optional target feature
        indices_t       m_selected;     ///< indices of the selected input features
    };

    using imodel_t = identifiable_t<model_t>;
    using imodels_t = std::vector<imodel_t>;

    ///
    /// \brief gather the results of k-fold cross-validation.
    ///
    struct kfold_result_t
    {
        kfold_result_t() = default;
        explicit kfold_result_t(tensor_size_t folds);

        tensor1d_t      m_train_errors; ///<
        tensor1d_t      m_valid_errors; ///<
        rmodels_t       m_models;       ///<
    };

    ///
    /// \brief (repeated) k-fold cross-validation
    ///     using the given model as currently setup in terms of (hyper-)parameters.
    ///
    NANO_PUBLIC kfold_result_t kfold(
        const model_t&, const dataset_generator_t&, const indices_t&, const loss_t& loss, const solver_t&,
        tensor_size_t folds = 5, tensor_size_t repetitions = 1);
}
