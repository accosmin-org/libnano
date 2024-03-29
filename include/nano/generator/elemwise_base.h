#pragma once

#include <nano/generator.h>

namespace nano
{
///
/// \brief interface for element-wise feature generators.
///
class NANO_PUBLIC base_elemwise_generator_t : public generator_t
{
public:
    ///
    /// \brief default constructor (use all available features).
    ///
    explicit base_elemwise_generator_t(string_t id);

    ///
    /// \brief constructor (use the given features, if of the appropriate type).
    ///
    explicit base_elemwise_generator_t(string_t id, indices_t features);

    ///
    /// \brief @see generator_t
    ///
    void fit(const datasource_t&) override;

    ///
    /// \brief @see generator_t
    ///
    tensor_size_t features() const override { return m_feature_mapping.size<0>(); }

protected:
    tensor3d_dims_t mapped_dims(tensor_size_t ifeature) const;
    tensor_size_t   mapped_classes(tensor_size_t ifeature) const;
    tensor_size_t   mapped_original(tensor_size_t ifeature) const;

    const feature_mapping_t& mapping() const { return m_feature_mapping; }

    const indices_t& original_features() const { return m_original_features; }

    feature_t make_scalar_feature(tensor_size_t ifeature, const char* name) const;
    feature_t make_sclass_feature(tensor_size_t ifeature, const char* name, strings_t labels) const;
    feature_t make_mclass_feature(tensor_size_t ifeature, const char* name, strings_t labels) const;
    feature_t make_struct_feature(tensor_size_t ifeature, const char* name, tensor3d_dims_t) const;

private:
    virtual feature_mapping_t do_fit() = 0;

    // attributes
    indices_t         m_original_features; ///< indices of the original features to use
    feature_mapping_t m_feature_mapping;   ///< (feature index, original feature index, ...)
};
} // namespace nano
