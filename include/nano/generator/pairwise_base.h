#pragma once

#include <nano/generator.h>

namespace nano
{
    ///
    /// \brief interface for pair-wise feature generators.
    ///
    class NANO_PUBLIC base_pairwise_generator_t : public generator_t
    {
    public:

        base_pairwise_generator_t();
        explicit base_pairwise_generator_t(indices_t original_features);
        base_pairwise_generator_t(indices_t original_features1, indices_t original_features2);

        void fit(const dataset_t& dataset) override;

        tensor_size_t features() const override
        {
            return m_feature_mapping.size<0>();
        }

    protected:

        tensor3d_dims_t mapped_dims1(tensor_size_t ifeature) const;
        tensor3d_dims_t mapped_dims2(tensor_size_t ifeature) const;
        tensor_size_t mapped_classes1(tensor_size_t ifeature) const;
        tensor_size_t mapped_classes2(tensor_size_t ifeature) const;
        tensor_size_t mapped_original1(tensor_size_t ifeature) const;
        tensor_size_t mapped_original2(tensor_size_t ifeature) const;

        const auto& mapping() const { return m_feature_mapping; }
        const auto& original_features1() const { return m_original_features1; }
        const auto& original_features2() const { return m_original_features2; }

        static feature_mapping_t make_pairwise(const feature_mapping_t& mapping1, const feature_mapping_t& mapping2);
        feature_t make_scalar_feature(tensor_size_t ifeature, const char* name) const;
        feature_t make_sclass_feature(tensor_size_t ifeature, const char* name, strings_t labels) const;
        feature_t make_mclass_feature(tensor_size_t ifeature, const char* name, strings_t labels) const;
        feature_t make_struct_feature(tensor_size_t ifeature, const char* name, tensor3d_dims_t) const;

    private:

        virtual feature_mapping_t do_fit() = 0;

        // attributes
        indices_t           m_original_features1;   ///< indices of the original features to use
        indices_t           m_original_features2;   ///< indices of the original features to use
        feature_mapping_t   m_feature_mapping;      ///< (feature index, original feature index, ...)
    };
}
