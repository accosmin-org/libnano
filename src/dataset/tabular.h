#pragma once

#include <nano/dataset.h>

namespace nano
{
    ///
    /// \brief create machine learning dataset from tabular data (CSV files).
    ///
    ///
    class tabular_dataset_t : public dataset_t
    {
    public:

        tabular_dataset_t() = default;
        tabular_dataset_t(std::vector<feature_t> features);

        bool load() override;

        json_t config() const override;
        void config(const json_t&) override;

        size_t folds() const override;
        void shuffle(const fold_t&) override;

        size_t ifeatures() const override;
        feature_t tfeature() const override;
        feature_t ifeature(const size_t) const override;

        tensor4d_t inputs(const fold_t&) const override;
        tensor4d_t targets(const fold_t&) const override;

    protected:

        ///
        /// \brief generate a split into training, validation and test.
        ///
        virtual void split(split_t&) const;

    private:

        auto samples() const { return m_inputs.size<0>(); }

        bool split();
        bool parse(const string_t& path, tensor_size_t& row_offset);

        void store(const tensor_size_t row, const size_t feature, const scalar_t value);
        void store(const tensor_size_t row, const size_t feature, const tensor_size_t category);

        indices_t& indices(const fold_t&);
        const indices_t& indices(const fold_t&) const;

        static tensor_size_t lines(const string_t& path);
        static tensor4d_t index(const tensor4d_t&, const indices_t&);

    private:

        // attributes
        tensor_size_t           m_train_per{80};///< percentage ot samples used for training (without the test samples)
        string_t                m_delim{","};   ///< CSV delimeter character
        strings_t               m_paths;        ///< CSV files to load one after the other
        size_t                  m_target{string_t::npos};///< index of the target column (if negative, then not provided)
        std::vector<feature_t>  m_features;     ///< describes all CSV columns
        std::vector<split_t>    m_splits;       ///<
        tensor4d_t              m_inputs;       ///< (total number of samples, number of inputs, 1, 1)
        tensor4d_t              m_targets;      ///< (total number of samples, number of outputs, 1, 1)
    };
}
