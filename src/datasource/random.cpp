#include <nano/datasource/random.h>

using namespace nano;

random_datasource_t::random_datasource_t(const tensor_size_t samples, features_t features, const size_t target,
                                         tensor_mem_t<int8_t, 2> hits)
    : datasource_t("random")
    , m_samples(samples)
    , m_features(std::move(features))
    , m_target(target)
    , m_hits(std::move(hits))
{
    assert(m_hits.rows() == m_samples);
    assert(m_hits.cols() == static_cast<tensor_size_t>(m_features.size()));

    register_parameter(parameter_t::make_integer("datasource::random::seed", 0, LE, 42, LE, 1024));
}

rdatasource_t random_datasource_t::clone() const
{
    return std::make_unique<random_datasource_t>(*this);
}

void random_datasource_t::do_load()
{
    const auto seed = parameter("datasource::random::seed").value<uint64_t>();

    datasource_t::resize(m_samples, m_features, m_target);

    tensor_size_t ifeature = 0;
    for (const auto& feature : m_features)
    {
        switch (feature.type())
        {
        case feature_type::sclass:
            set_fvalues(ifeature, make_random_tensor<int8_t>(make_dims(m_samples), tensor_size_t{0},
                                                             feature.classes() - 1, seed));
            break;

        case feature_type::mclass:
            set_fvalues(ifeature, make_random_tensor<int8_t>(make_dims(m_samples, feature.classes()), 0, 1, seed));
            break;

        case feature_type::uint8:
        case feature_type::uint16:
        case feature_type::uint32:
        case feature_type::uint64:
            set_fvalues(ifeature, make_random_tensor<uint8_t>(cat_dims(m_samples, feature.dims()), 0, 13, seed));
            break;

        case feature_type::int8:
        case feature_type::int16:
        case feature_type::int32:
        case feature_type::int64:
            set_fvalues(ifeature, make_random_tensor<int8_t>(cat_dims(m_samples, feature.dims()), -11, +17, seed));
            break;

        case feature_type::float32:
            set_fvalues(ifeature, make_random_tensor<float>(cat_dims(m_samples, feature.dims()), -3.0, +2.9, seed));
            break;

        default:
            set_fvalues(ifeature, make_random_tensor<scalar_t>(cat_dims(m_samples, feature.dims()), -1.2, +1.3, seed));
            break;
        }
        ++ifeature;
    }
}
