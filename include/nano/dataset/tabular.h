#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief stores information required to load a CSV file.
    ///
    struct csv_t
    {
        csv_t() = default;
        explicit csv_t(string_t path) : m_path(std::move(path)) {}

        auto& skip(const char skip) { m_skip = skip; return *this; }
        auto& header(const bool header) { m_header = header; return *this; }
        auto& delim(string_t delim) { m_delim = std::move(delim); return *this; }
        auto& expected(const int expected) { m_expected = expected; return *this; }

        string_t    m_path;             ///<
        string_t    m_delim{", \r"};    ///< delimiting characters
        char        m_skip{'#'};        ///< skip lines starting with this character
        bool        m_header{false};    ///< skip the first line with the header
        int         m_expected{-1};     ///< expected number of lines to read (excepting skipped lines and the header)
    };

    ///
    /// \brief interface to create machine learning datasets from tabular data (CSV files).
    ///
    /// NB: the customization point (in the derived classes)
    ///     consists of generating the training, validation and test dataset splits.
    ///
    class NANO_PUBLIC tabular_dataset_t : public dataset_t
    {
    public:

        tabular_dataset_t() = default;

        bool load() override;

        size_t folds() const override;
        void shuffle(const fold_t&) override;

        size_t ifeatures() const override;
        feature_t tfeature() const override;
        feature_t ifeature(const size_t) const override;

        tensor4d_t inputs(const fold_t&) const override;
        tensor4d_t targets(const fold_t&) const override;

        ///
        /// \brief generate a split into training, validation and test.
        ///
        virtual void split(const tensor_size_t samples, split_t&) const = 0;

        ///
        /// \brief setup tabular dataset (e.g. csv paths, delimeter string, folds, features)
        ///
        void csvs(std::vector<csv_t>);
        void folds(const size_t folds);
        void features(std::vector<feature_t>, const size_t target = string_t::npos);

    private:

        auto samples() const { return m_inputs.size<0>(); }

        bool parse(const string_t& path, const string_t& line, const string_t& delim,
            const tensor_size_t line_index, const tensor_size_t row);

        void store(const tensor_size_t row, const size_t feature, const scalar_t value);
        void store(const tensor_size_t row, const size_t feature, const tensor_size_t category);

        indices_t& indices(const fold_t&);
        const indices_t& indices(const fold_t&) const;

    private:

        // attributes
        std::vector<csv_t>      m_csvs;         ///<
        size_t                  m_target{string_t::npos};///< index of the target column (if negative, then not provided)
        std::vector<feature_t>  m_features;     ///< describes all CSV columns
        std::vector<split_t>    m_splits{10};   ///<
        tensor4d_t              m_inputs;       ///< (total number of samples, number of inputs, 1, 1)
        tensor4d_t              m_targets;      ///< (total number of samples, number of outputs, 1, 1)
    };
}
