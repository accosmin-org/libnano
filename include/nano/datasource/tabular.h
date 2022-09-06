#pragma once

#include <nano/datasource.h>
#include <nano/datasource/csv.h>

namespace nano
{
    ///
    /// \brief machine learning dataset consisting of samples loaded from CSV files (aka tabular data).
    ///
    /// the tabular dataset is versatile:
    ///     - the target is optional, so it can address both supervised and unsupervised machine learning tasks
    ///     - the inputs can be both categorical and continuous
    ///     - missing feature values are supported
    ///
    class NANO_PUBLIC tabular_datasource_t : public datasource_t
    {
    public:
        ///
        /// \brief constructor, set the CSV files to load and the input features.
        ///
        tabular_datasource_t(string_t id, csvs_t, features_t);

        ///
        /// \brief constructor, set the CSV files to load and the input and the target features.
        ///
        tabular_datasource_t(string_t id, csvs_t, features_t, size_t target);

        ///
        /// \brief @see clonable_t
        ///
        rdatasource_t clone() const override;

    private:
        void do_load() override;
        void parse(const csv_t&, const string_t&, tensor_size_t, tensor_size_t);

        // attributes
        csvs_t     m_csvs;                   ///< describes the CSV files
        features_t m_features;               ///< describes the columns in the CSV files (aka the features)
        size_t     m_target{string_t::npos}; ///< index of the target column (if negative, then not provided)
    };
} // namespace nano
