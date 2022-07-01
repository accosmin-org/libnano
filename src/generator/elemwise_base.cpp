#include <nano/generator/elemwise_base.h>

using namespace nano;

base_elemwise_generator_t::base_elemwise_generator_t() = default;

base_elemwise_generator_t::base_elemwise_generator_t(indices_t original_features)
    : m_original_features(std::move(original_features))
{
}

void base_elemwise_generator_t::fit(const dataset_t& dataset)
{
    generator_t::fit(dataset);
    m_feature_mapping = do_fit();
    allocate(features());
}

tensor_size_t base_elemwise_generator_t::mapped_original(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return m_feature_mapping(ifeature, 0);
}

tensor_size_t base_elemwise_generator_t::mapped_classes(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return m_feature_mapping(ifeature, 1);
}

tensor3d_dims_t base_elemwise_generator_t::mapped_dims(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return make_dims(m_feature_mapping(ifeature, 2), m_feature_mapping(ifeature, 3), m_feature_mapping(ifeature, 4));
}

feature_t base_elemwise_generator_t::make_scalar_feature(tensor_size_t ifeature, const char* name) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original = mapped_original(ifeature);

    const auto& feature = dataset().feature(original);
    return feature_t{scat(name, "(", feature.name(), ")")}.scalar(feature_type::float64);
}

feature_t base_elemwise_generator_t::make_sclass_feature(tensor_size_t ifeature, const char* name,
                                                         strings_t labels) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original = mapped_original(ifeature);

    const auto& feature = dataset().feature(original);
    return feature_t{scat(name, "(", feature.name(), ")")}.sclass(std::move(labels));
}

feature_t base_elemwise_generator_t::make_mclass_feature(tensor_size_t ifeature, const char* name,
                                                         strings_t labels) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original = mapped_original(ifeature);

    const auto& feature = dataset().feature(original);
    return feature_t{scat(name, "(", feature.name(), ")")}.mclass(std::move(labels));
}

feature_t base_elemwise_generator_t::make_struct_feature(tensor_size_t ifeature, const char* name,
                                                         tensor3d_dims_t dims) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original = mapped_original(ifeature);

    const auto& feature = dataset().feature(original);
    return feature_t{scat(name, "(", feature.name(), ")")}.scalar(feature_type::float64, dims);
}
