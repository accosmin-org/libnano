#include <nano/dataset/iterator.h>
#include <nano/datasource/linear.h>
#include <nano/generator/elemwise_identity.h>

using namespace nano;

namespace
{
features_t make_features(const tensor_size_t n_features, const tensor_size_t n_targets)
{
    features_t features;
    for (tensor_size_t ifeature = 0; ifeature < n_features; ++ifeature)
    {
        feature_t feature;
        switch (ifeature % 4)
        {
        case 0: feature = feature_t{scat("scalar", ifeature)}.scalar(); break;
        case 1: feature = feature_t{scat("sclass", ifeature)}.sclass(3U); break;
        case 2: feature = feature_t{scat("mclass", ifeature)}.mclass(4U); break;
        default: feature = feature_t{scat("struct", ifeature)}.scalar(feature_type::float64, make_dims(2, 1, 3)); break;
        }
        features.push_back(feature);
    }
    features.push_back(feature_t{"Wx+b+eps"}.scalar(feature_type::float64, make_dims(n_targets, 1, 1)));

    return features;
}

auto make_dataset(const datasource_t& datasource)
{
    auto dataset = dataset_t{datasource};
    dataset.add<elemwise_generator_t<sclass_identity_t>>();
    dataset.add<elemwise_generator_t<mclass_identity_t>>();
    dataset.add<elemwise_generator_t<scalar_identity_t>>();
    dataset.add<elemwise_generator_t<struct_identity_t>>();
    return dataset;
}
} // namespace

linear_datasource_t::linear_datasource_t()
    : datasource_t("synth-linear")
{
    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_integer("datasource::linear::seed", 0, LE, 42, LE, 1024));
    register_parameter(parameter_t::make_integer("datasource::linear::modulo", 1, LE, 3, LE, 1000));
    register_parameter(parameter_t::make_integer("datasource::linear::samples", 10, LE, 100, LE, 1e+6));
    register_parameter(parameter_t::make_integer("datasource::linear::features", 1, LE, 1, LE, 1e+6));
    register_parameter(parameter_t::make_integer("datasource::linear::targets", 1, LE, 1, LE, 1e+3));

    register_parameter(parameter_t::make_scalar("datasource::linear::noise", 0.0, LE, 0.0, LE, fmax));
    register_parameter(parameter_t::make_scalar("datasource::linear::missing_percentage", 0, LE, 0, LE, 100));
}

rdatasource_t linear_datasource_t::clone() const
{
    return std::make_unique<linear_datasource_t>(*this);
}

void linear_datasource_t::do_load()
{
    const auto seed      = parameter("datasource::linear::seed").value<uint64_t>();
    const auto modulo    = parameter("datasource::linear::modulo").value<tensor_size_t>();
    const auto samples   = parameter("datasource::linear::samples").value<tensor_size_t>();
    const auto nfeatures = parameter("datasource::linear::features").value<tensor_size_t>();
    const auto ntargets  = parameter("datasource::linear::targets").value<tensor_size_t>();
    const auto noise     = parameter("datasource::linear::noise").value<scalar_t>();
    const auto missing   = parameter("datasource::linear::missing_percentage").value<int>();

    const auto features = ::make_features(nfeatures, ntargets);
    const auto itarget  = features.size() - 1U;
    datasource_t::resize(samples, features, itarget);

    auto rng   = make_rng(seed);
    auto mdist = make_udist<int>(0, 99);
    auto sdist = make_udist<uint64_t>(0, 1024);

    const auto hitter = [&]() { return mdist(rng) >= missing; };

    tensor_size_t ifeature = 0;
    for (const auto& feature : features)
    {
        switch (feature.type())
        {
        case feature_type::sclass:
            setter(ifeature,
                   make_random_tensor<tensor_size_t>(make_dims(samples), tensor_size_t{0}, feature.classes() - 1,
                                                     sdist(rng)),
                   hitter);
            break;

        case feature_type::mclass:
            setter(ifeature, make_random_tensor<int8_t>(make_dims(samples, feature.classes()), 0, 1, sdist(rng)),
                   hitter);
            break;

        case feature_type::uint8:
        case feature_type::uint16:
        case feature_type::uint32:
        case feature_type::uint64:
            setter(ifeature, make_random_tensor<uint8_t>(cat_dims(samples, feature.dims()), 0, 13, sdist(rng)), hitter);
            break;

        case feature_type::int8:
        case feature_type::int16:
        case feature_type::int32:
        case feature_type::int64:
            setter(ifeature, make_random_tensor<int8_t>(cat_dims(samples, feature.dims()), -11, +17, sdist(rng)),
                   hitter);
            break;

        case feature_type::float32:
            setter(ifeature, make_random_tensor<float>(cat_dims(samples, feature.dims()), -3.0, +2.9, sdist(rng)),
                   hitter);
            break;

        default:
            setter(ifeature, make_random_tensor<scalar_t>(cat_dims(samples, feature.dims()), -1.2, +1.3, sdist(rng)),
                   hitter);
            break;
        }
        ++ifeature;
    }

    // create samples: target = weights * input + bias + noise
    const auto dataset = ::make_dataset(*this);
    const auto targets = ::nano::size(dataset.target_dims());

    m_bias    = make_random_tensor<scalar_t>(make_dims(targets), -1.0, +1.0, sdist(rng));
    m_weights = make_random_tensor<scalar_t>(make_dims(targets, dataset.columns()), -1.0, +1.0, sdist(rng));

    for (tensor_size_t column = 0, columns = dataset.columns(); column < columns; ++column)
    {
        const auto feature = dataset.column2feature(column);
        if (feature % modulo == 0)
        {
            m_weights.matrix().col(column).setConstant(0.0);
        }
    }

    auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
    iterator.loop(
        [&](tensor_range_t range, size_t, tensor2d_cmap_t inputs)
        {
            auto target  = tensor1d_t{targets};
            auto weights = m_weights.matrix();

            for (tensor_size_t i = 0, size = range.size(); i < size; ++i)
            {
                target.vector() = weights * inputs.vector(i) + m_bias.vector();
                target.vector() += noise * make_random_vector<scalar_t>(m_bias.size());
                set(i + range.begin(), static_cast<tensor_size_t>(itarget), target);
            }
        });
}