#pragma once

#include <nano/dataset.h>

namespace nano
{
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
        void skip(char);
        void delim(string_t);
        void paths(strings_t paths);
        void folds(const size_t folds);
        void features(std::vector<feature_t>, const size_t target = string_t::npos);

    private:

        auto samples() const { return m_inputs.size<0>(); }

        bool parse(const string_t& path, tensor_size_t& row_offset);

        void store(const tensor_size_t row, const size_t feature, const scalar_t value);
        void store(const tensor_size_t row, const size_t feature, const tensor_size_t category);

        indices_t& indices(const fold_t&);
        const indices_t& indices(const fold_t&) const;

    private:

        // attributes
        char                    m_skip{'#'};    ///< CSV character for lines to ignore
        string_t                m_delim{","};   ///< CSV delimeter character
        strings_t               m_paths;        ///< CSV files to load one after the other
        size_t                  m_target{string_t::npos};///< index of the target column (if negative, then not provided)
        std::vector<feature_t>  m_features;     ///< describes all CSV columns
        std::vector<split_t>    m_splits{10};   ///<
        tensor4d_t              m_inputs;       ///< (total number of samples, number of inputs, 1, 1)
        tensor4d_t              m_targets;      ///< (total number of samples, number of outputs, 1, 1)
    };
}
