#pragma once

#include <nano/arch.h>
#include <nano/factory.h>
#include <nano/memfixed.h>
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
    class NANO_PUBLIC tabular_dataset_t : public memfixed_dataset_t<scalar_t>
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
        ~tabular_dataset_t() override = default;

        ///
        /// \brief populate the dataset with samples
        ///
        bool load() override;

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
        feature_t tfeature() const override;

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
        size_t      m_target{string_t::npos};   ///< index of the target column (if negative, then not provided)
    };
}
