#pragma once

#include <nano/arch.h>
#include <nano/factory.h>
#include <nano/feature.h>
#include <nano/dataset.h>
#include <nano/tabular/csv.h>

namespace nano
{
    class tabular_dataset_t;
    using tabular_dataset_factory_t = factory_t<tabular_dataset_t>;
    using rtabular_dataset_t = tabular_dataset_factory_t::trobject;

    ///
    /// \brief machine learning dataset consisting of samples loaded from CSV files (aka tabular data).
    ///
    /// the tabular dataset is versatile:
    ///     - the target is optional, so it can address both supervised and unsupervised machine learning tasks
    ///     - the inputs can be both categorical and continuous
    ///     - missing feature values are supported
    ///
    class NANO_PUBLIC tabular_dataset_t : public dataset_t
    {
    public:

        ///
        /// \brief returns the available implementations
        ///
        static tabular_dataset_factory_t& all();

        ///
        /// \brief default constructor
        ///
        tabular_dataset_t() = default;

        ///
        /// \brief enable copying
        ///
        tabular_dataset_t(const tabular_dataset_t&) = default;
        tabular_dataset_t& operator=(const tabular_dataset_t&) = default;

        ///
        /// \brief enable moving
        ///
        tabular_dataset_t(tabular_dataset_t&&) noexcept = default;
        tabular_dataset_t& operator=(tabular_dataset_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        virtual ~tabular_dataset_t() = default;

        ///
        /// \brief populate the dataset with samples
        ///
        bool load();

        ///
        /// \brief returns the total number of samples
        ///
        tensor_size_t samples() const;

        ///
        /// \brief returns the number of samples associated to a given fold
        ///
        tensor_size_t samples(const fold_t&) const;

        ///
        /// \brief returns the total number of input features
        ///
        size_t ifeatures() const;

        ///
        /// \brief returns the description of the given input feature
        ///
        feature_t ifeature(size_t index) const;

        ///
        /// \brief returns the description of the target feature (if a supervised task)
        ///
        feature_t tfeature() const;

        ///
        /// \brief returns the inputs tensor for all samples in the given fold
        ///
        tensor4d_t inputs(const fold_t&) const;

        ///
        /// \brief returns the inputs tensor for the [begin, end) range of samples in the given fold
        ///
        tensor4d_t inputs(const fold_t&, tensor_size_t begin, tensor_size_t end) const;

        ///
        /// \brief returns the targets tensor for all samples in the given fold (if a supervised task)
        ///
        tensor4d_t targets(const fold_t&) const;

        ///
        /// \brief returns the targets tensor for the [begin, end) range of samples in the given fold (if a supervised task)
        ///
        tensor4d_t targets(const fold_t&, tensor_size_t begin, tensor_size_t end) const;

        ///
        /// \brief set the CSV files to load
        ///
        void csvs(std::vector<csv_t>);

        ///
        /// \brief set the input and the target features
        ///
        void features(std::vector<feature_t>, size_t target = string_t::npos);

        ///
        /// \brief generate a split into training, validation and test.
        ///
        virtual split_t make_split() const = 0;

    protected:

        void store(tensor_size_t row, size_t feature, scalar_t value);
        void store(tensor_size_t row, size_t feature, tensor_size_t category);
        bool parse(const string_t&, const string_t&, const string_t&, tensor_size_t, tensor_size_t);

    private:

        // attributes
        csvs_t      m_csvs;                     ///< describes the CSV files
        features_t  m_features;                 ///< describes the columns in the CSV files (aka the features)
        tensor4d_t  m_inputs;                   ///< (total number of samples, number of inputs, 1, 1)
        tensor4d_t  m_targets;                  ///< (total number of samples, number of outputs, 1, 1)
        size_t      m_target{string_t::npos};   ///< index of the target column (if negative, then not provided)
    };
}
