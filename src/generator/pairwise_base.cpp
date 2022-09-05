#include <nano/generator/pairwise.h>

using namespace nano;

base_pairwise_generator_t::base_pairwise_generator_t(string_t id)
    : generator_t(std::move(id))
{
}

base_pairwise_generator_t::base_pairwise_generator_t(string_t id, indices_t original_features)
    : generator_t(std::move(id))
    , m_original_features1(original_features)
    , m_original_features2(std::move(original_features))
{
}

base_pairwise_generator_t::base_pairwise_generator_t(string_t id, indices_t original_features1,
                                                     indices_t original_features2)
    : generator_t(std::move(id))
    , m_original_features1(std::move(original_features1))
    , m_original_features2(std::move(original_features2))
{
}

void base_pairwise_generator_t::fit(const dataset_t& dataset)
{
    generator_t::fit(dataset);
    m_feature_mapping = do_fit();
    allocate(features());
}

tensor_size_t base_pairwise_generator_t::mapped_original1(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return m_feature_mapping(ifeature, 0);
}

tensor_size_t base_pairwise_generator_t::mapped_original2(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return m_feature_mapping(ifeature, 5);
}

tensor_size_t base_pairwise_generator_t::mapped_classes1(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return m_feature_mapping(ifeature, 1);
}

tensor_size_t base_pairwise_generator_t::mapped_classes2(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return m_feature_mapping(ifeature, 6);
}

tensor3d_dims_t base_pairwise_generator_t::mapped_dims1(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return make_dims(m_feature_mapping(ifeature, 2), m_feature_mapping(ifeature, 3), m_feature_mapping(ifeature, 4));
}

tensor3d_dims_t base_pairwise_generator_t::mapped_dims2(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return make_dims(m_feature_mapping(ifeature, 7), m_feature_mapping(ifeature, 8), m_feature_mapping(ifeature, 9));
}

feature_mapping_t base_pairwise_generator_t::make_pairwise(const feature_mapping_t& mapping1,
                                                           const feature_mapping_t& mapping2)
{
    const auto size1 = mapping1.template size<0>();
    const auto size2 = mapping2.template size<0>();

    const auto vals1 = mapping1.template size<1>();
    const auto vals2 = mapping2.template size<1>();

    const auto combine = [=](feature_mapping_t& mapping, tensor_size_t k, tensor_size_t i1, tensor_size_t i2)
    {
        mapping.array(k).segment(0, vals1)     = mapping1.array(i1);
        mapping.array(k).segment(vals1, vals2) = mapping2.array(i2);
    };

    std::map<std::pair<tensor_size_t, tensor_size_t>, std::pair<tensor_size_t, tensor_size_t>> upairs;
    for (tensor_size_t i1 = 0; i1 < size1; ++i1)
    {
        for (tensor_size_t i2 = 0; i2 < size2; ++i2)
        {
            const auto feature1 = mapping1(i1, 0);
            const auto feature2 = mapping2(i2, 0);

            const auto key   = std::make_pair(std::min(feature1, feature2), std::max(feature1, feature2));
            const auto value = (feature1 <= feature2) ? std::make_pair(i1, i2) : std::make_pair(i2, i1);
            upairs.try_emplace(key, value);
        }
    }

    auto feature_mapping = feature_mapping_t{static_cast<tensor_size_t>(upairs.size()), vals1 + vals2};

    tensor_size_t k = 0;
    for (const auto& upair : upairs)
    {
        const auto [i1, i2]                              = upair.second;
        feature_mapping.array(k).segment(0, vals1)       = mapping1.array(i1);
        feature_mapping.array(k++).segment(vals1, vals2) = mapping2.array(i2);
    }

    return feature_mapping;
}

feature_t base_pairwise_generator_t::make_scalar_feature(tensor_size_t ifeature, const char* name) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original1 = mapped_original1(ifeature);
    const auto original2 = mapped_original2(ifeature);

    const auto& feature1 = dataset().feature(original1);
    const auto& feature2 = dataset().feature(original2);

    return feature_t{scat(name, "(", feature1.name(), ",", feature2.name(), ")")}.scalar(feature_type::float64);
}

feature_t base_pairwise_generator_t::make_sclass_feature(tensor_size_t ifeature, const char* name,
                                                         strings_t labels) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original1 = mapped_original1(ifeature);
    const auto original2 = mapped_original2(ifeature);

    const auto& feature1 = dataset().feature(original1);
    const auto& feature2 = dataset().feature(original2);

    return feature_t{scat(name, "(", feature1.name(), ",", feature2.name(), ")")}.sclass(std::move(labels));
}

feature_t base_pairwise_generator_t::make_mclass_feature(tensor_size_t ifeature, const char* name,
                                                         strings_t labels) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original1 = mapped_original1(ifeature);
    const auto original2 = mapped_original2(ifeature);

    const auto& feature1 = dataset().feature(original1);
    const auto& feature2 = dataset().feature(original2);

    return feature_t{scat(name, "(", feature1.name(), ",", feature2.name(), ")")}.mclass(std::move(labels));
}

feature_t base_pairwise_generator_t::make_struct_feature(tensor_size_t ifeature, const char* name,
                                                         tensor3d_dims_t dims) const
{
    assert(ifeature >= 0 && ifeature < features());
    const auto original1 = mapped_original1(ifeature);
    const auto original2 = mapped_original2(ifeature);

    const auto& feature1 = dataset().feature(original1);
    const auto& feature2 = dataset().feature(original2);

    return feature_t{scat(name, "(", feature1.name(), ",", feature2.name(), ")")}.scalar(feature_type::float64, dims);
}
