#include <fixture/dataset.h>
#include <fixture/datasource/linear.h>
#include <nano/dataset/stats.h>

using namespace nano;

namespace
{
auto make_features()
{
    return features_t{
        feature_t{"sclass0"}.sclass(strings_t{"s10", "s11"}),
        feature_t{"mclass0"}.mclass(strings_t{"m00", "m01", "m02"}),
        feature_t{"scalar0"}.scalar(feature_type::float32),
        feature_t{"struct0"}.scalar(feature_type::float64, make_dims(1, 2, 2)),
    };
}

template <class... tvalues>
auto make_scalars(const tvalues... values)
{
    const auto samples = static_cast<tensor_size_t>(sizeof...(values));
    return make_tensor<scalar_t>(make_dims(samples), values...);
}

template <class... tvalues>
auto make_sclass_data(const tvalues... values)
{
    const auto samples = static_cast<tensor_size_t>(sizeof...(values));
    return make_tensor<int32_t>(make_dims(samples), values...);
}

template <class... tvalues>
auto make_mclass_data(const tvalues... values)
{
    const auto samples = static_cast<tensor_size_t>(sizeof...(values)) / 3;
    return make_tensor<int8_t>(make_dims(samples, 3), values...);
}

template <class... tvalues>
auto make_scalar_data(const tvalues... values)
{
    const auto samples = static_cast<tensor_size_t>(sizeof...(values));
    return make_tensor<scalar_t>(make_dims(samples), values...);
}

template <class... tvalues>
auto make_struct_data(const tvalues... values)
{
    const auto samples = static_cast<tensor_size_t>(sizeof...(values)) / 4;
    return make_tensor<scalar_t>(make_dims(samples, 1, 2, 2), values...);
}

class stats_datasource_t final : public datasource_t
{
public:
    stats_datasource_t(sclass_mem_t sclass_data, mclass_mem_t mclass_data, scalar_mem_t scalar_data,
                       struct_mem_t struct_data, const size_t target)
        : datasource_t("fixture")
        , m_sclass_data(std::move(sclass_data))
        , m_mclass_data(std::move(mclass_data))
        , m_scalar_data(std::move(scalar_data))
        , m_struct_data(std::move(struct_data))
        , m_target(target)
    {
        assert(m_sclass_data.size<0>() == m_mclass_data.size<0>());
        assert(m_sclass_data.size<0>() == m_scalar_data.size<0>());
        assert(m_sclass_data.size<0>() == m_struct_data.size<0>());
    }

    rdatasource_t clone() const override { return std::make_unique<stats_datasource_t>(*this); }

    const auto& sclass_data() const { return m_sclass_data; }

    const auto& mclass_data() const { return m_mclass_data; }

private:
    void do_load() override
    {
        const auto samples = m_sclass_data.size();

        resize(samples, make_features(), m_target);

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            const auto value = m_sclass_data(sample);
            if (value >= 0)
            {
                set(sample, 0, value);
            }
        }

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            const auto value = m_mclass_data.tensor(sample);
            if (value(0) >= 0)
            {
                set(sample, 1, value);
            }
        }

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            const auto value = m_scalar_data(sample);
            if (std::isfinite(value))
            {
                set(sample, 2, value);
            }
        }

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            const auto value = m_struct_data.tensor(sample);
            if (std::isfinite(value(0)))
            {
                set(sample, 3, value);
            }
        }
    }

    sclass_mem_t m_sclass_data;
    mclass_mem_t m_mclass_data;
    scalar_mem_t m_scalar_data;
    struct_mem_t m_struct_data;
    size_t       m_target{0U};
};

void check_xclass_stats(const xclass_stats_t& stats, const hashes_t& expected_class_hashes,
                        const indices_t& expected_class_samples, const indices_t& expected_sample_classes,
                        const tensor1d_t& expected_sample_weights, const scalar_t epsilon = 1e-12)
{
    UTEST_CHECK_EQUAL(stats.m_class_hashes, expected_class_hashes);
    UTEST_CHECK_EQUAL(stats.m_class_samples, expected_class_samples);
    UTEST_CHECK_EQUAL(stats.m_sample_classes, expected_sample_classes);
    UTEST_CHECK_CLOSE(stats.m_sample_weights, expected_sample_weights, epsilon);
}

void check_scalar_stats(const scalar_stats_t& stats, const indices_t& expected_samples, const tensor1d_t& expected_min,
                        const tensor1d_t& expected_max, const tensor1d_t& expected_mean,
                        const tensor1d_t& expected_stdev, const tensor1d_t& expected_div_range,
                        const tensor1d_t& expected_mul_range, const tensor1d_t& expected_div_stdev,
                        const tensor1d_t& expected_mul_stdev, const scalar_t epsilon = 1e-12)
{
    UTEST_CHECK_EQUAL(stats.m_samples, expected_samples);
    UTEST_CHECK_CLOSE(stats.m_min, expected_min, epsilon);
    UTEST_CHECK_CLOSE(stats.m_max, expected_max, epsilon);
    UTEST_CHECK_CLOSE(stats.m_mean, expected_mean, epsilon);
    UTEST_CHECK_CLOSE(stats.m_stdev, expected_stdev, epsilon);
    UTEST_CHECK_CLOSE(stats.m_div_range, expected_div_range, epsilon);
    UTEST_CHECK_CLOSE(stats.m_mul_range, expected_mul_range, epsilon);
    UTEST_CHECK_CLOSE(stats.m_div_stdev, expected_div_stdev, epsilon);
    UTEST_CHECK_CLOSE(stats.m_mul_stdev, expected_mul_stdev, epsilon);
}

auto make_stats_datasource(const size_t target)
{
    auto sclass_data = make_sclass_data(0, 1, -1, 0, 0, 1);
    auto mclass_data = make_mclass_data(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1);
    auto scalar_data = make_scalar_data(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    auto struct_data = make_struct_data(INF, INF, INF, INF, 1.0, 10.0, -1.0, 0.0, 2.0, 20.0, -1.0, 0.0, 3.0, 30.0, -2.0,
                                        0.0, 5.0, 40.0, -3.0, 0.0, 4.0, 50.0, -3.0, 1.0);

    switch (target)
    {
    case 0U:
        sclass_data(2) = 0;
        break;
    case 3U:
        struct_data(0) = 0;
        struct_data(1) = 0;
        struct_data(2) = 0;
        struct_data(3) = 0;
        break;
    default:
        break;
    }

    auto datasource = stats_datasource_t{std::move(sclass_data), std::move(mclass_data), std::move(scalar_data),
                                         std::move(struct_data), target};
    UTEST_REQUIRE_NOTHROW(datasource.load());
    return datasource;
}

auto make_flatten(const dataset_t& dataset, const indices_t& samples)
{
    tensor2d_t buffer;
    dataset.flatten(samples, buffer);
    return buffer;
}

auto make_targets(const dataset_t& dataset, const indices_t& samples)
{
    tensor4d_t buffer;
    dataset.targets(samples, buffer);
    return buffer;
}
} // namespace

UTEST_BEGIN_MODULE(test_dataset_stats)

UTEST_CASE(scaling)
{
    const auto datasource = make_stats_datasource(string_t::npos);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    auto flatten = make_flatten(dataset, samples);
    for (auto& value : flatten)
    {
        if (!std::isfinite(value))
        {
            value = 0.0;
        }
    }

    const auto stats = scalar_stats_t::make_flatten_stats(dataset, samples);

    for (const auto scaling : enum_values<scaling_type>())
    {
        tensor2d_t values = flatten;
        UTEST_CHECK_NOTHROW(stats.scale(scaling, values));
        UTEST_CHECK_NOTHROW(stats.upscale(scaling, values));
        UTEST_CHECK_CLOSE(values, flatten, 1e-12);
    }
    {
        tensor2d_t values = flatten;
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        UTEST_CHECK_THROW(stats.scale(static_cast<scaling_type>(0xFF), values), std::runtime_error);
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        UTEST_CHECK_THROW(stats.upscale(static_cast<scaling_type>(0xFF), values), std::runtime_error);
        UTEST_CHECK_CLOSE(values, flatten, 1e-12);
    }
}

UTEST_CASE(scaling2)
{
    const auto datasource = make_linear_datasource(100, 13, 5);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    const auto flatten_stats = scalar_stats_t::make_flatten_stats(dataset, samples);
    const auto targets_stats = scalar_stats_t::make_targets_stats(dataset, samples);

    // check upscaling of affine transformations
    for (const auto flatten_scaling : enum_values<scaling_type>())
    {
        for (const auto targets_scaling : enum_values<scaling_type>())
        {
            auto bias    = datasource.bias();
            auto weights = datasource.weights();
            auto flatten = make_flatten(dataset, samples);
            auto targets = make_targets(dataset, samples);

            flatten_stats.upscale(flatten_scaling, flatten);
            targets_stats.upscale(targets_scaling, targets);
            ::nano::upscale(flatten_stats, flatten_scaling, targets_stats, targets_scaling, weights, bias);

            for (tensor_size_t i = 0; i < samples.size(); ++i)
            {
                UTEST_CHECK_CLOSE(weights.matrix() * flatten.vector(i) + bias.vector(), targets.vector(i), 1e-12);
            }
        }
    }
}

UTEST_CASE(no_target)
{
    const auto datasource = make_stats_datasource(string_t::npos);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    const auto sclass_hashes = make_hashes(datasource.sclass_data());
    const auto mclass_hashes = make_hashes(datasource.mclass_data());

    UTEST_CHECK_THROW(scalar_stats_t::make_targets_stats(dataset, samples), std::runtime_error);
    UTEST_CHECK_THROW(xclass_stats_t::make_targets_stats(dataset, samples), std::runtime_error);

    UTEST_CHECK_THROW(scalar_stats_t::make_feature_stats(dataset, samples, 0), std::runtime_error);
    UTEST_CHECK_THROW(scalar_stats_t::make_feature_stats(dataset, samples, 1), std::runtime_error);
    UTEST_CHECK_THROW(xclass_stats_t::make_feature_stats(dataset, samples, 2), std::runtime_error);
    UTEST_CHECK_THROW(xclass_stats_t::make_feature_stats(dataset, samples, 3), std::runtime_error);

    for (const auto batch : {1, 2, 3, 10})
    {
        const auto [st0, st1, st2, st3] = std::make_tuple(1.5811388300842, 15.811388300842, 1.0, 0.44721359549996);

        check_xclass_stats(xclass_stats_t::make_feature_stats(dataset, samples, 0), sclass_hashes, make_indices(3, 2),
                           make_indices(0, 1, -1, 0, 0, 1), make_scalars(0.4, 0.6, 0.0, 0.4, 0.4, 0.6));

        check_xclass_stats(xclass_stats_t::make_feature_stats(dataset, samples, 1), mclass_hashes,
                           make_indices(2, 2, 1, 1), make_indices(1, 0, 0, 1, 2, 3),
                           make_scalars(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0));

        check_scalar_stats(scalar_stats_t::make_feature_stats(dataset, samples, 2, batch), make_indices(6),
                           make_scalars(1.0), make_scalars(1.0), make_scalars(1.0), make_scalars(0.0),
                           make_scalars(1e+8), make_scalars(1e-8), make_scalars(1e+8), make_scalars(1e-8));

        check_scalar_stats(scalar_stats_t::make_feature_stats(dataset, samples, 3, batch), make_indices(5, 5, 5, 5),
                           make_scalars(1.0, 10.0, -3.0, 0.0), make_scalars(5.0, 50.0, -1.0, 1.0),
                           make_scalars(3.0, 30.0, -2.0, 0.2), make_scalars(st0, st1, st2, st3),
                           make_scalars(1.0 / 4.0, 1.0 / 40.0, 1.0 / 2.0, 1.0 / 1.0), make_scalars(4.0, 40.0, 2.0, 1.0),
                           make_scalars(1.0 / st0, 1.0 / st1, 1.0 / st2, 1.0 / st3), make_scalars(st0, st1, st2, st3));

        check_scalar_stats(scalar_stats_t::make_flatten_stats(dataset, samples, batch),
                           make_indices(5, 6, 6, 6, 6, 5, 5, 5, 5),
                           make_scalars(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, -3.0, 0.0),
                           make_scalars(0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 50.0, -1.0, 1.0),
                           make_scalars(0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 30.0, -2.0, 0.2),
                           make_scalars(0.0, 0.0, 0.0, 0.0, 0.0, st0, st1, st2, st3),
                           make_scalars(1.0, 1.0, 1.0, 1.0, 1e+8, 1.0 / 4.0, 1.0 / 40.0, 1.0 / 2.0, 1.0 / 1.0),
                           make_scalars(1.0, 1.0, 1.0, 1.0, 1e-8, 4.0, 40.0, 2.0, 1.0),
                           make_scalars(1.0, 1.0, 1.0, 1.0, 1e+8, 1.0 / st0, 1.0 / st1, 1.0 / st2, 1.0 / st3),
                           make_scalars(1.0, 1.0, 1.0, 1.0, 1e-8, st0, st1, st2, st3));
    }
}

UTEST_CASE(target_sclass)
{
    const auto datasource = make_stats_datasource(0U);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    const auto sclass_hashes = make_hashes(datasource.sclass_data());
    const auto mclass_hashes = make_hashes(datasource.mclass_data());

    UTEST_CHECK_THROW(scalar_stats_t::make_feature_stats(dataset, samples, 0), std::runtime_error);
    UTEST_CHECK_THROW(xclass_stats_t::make_feature_stats(dataset, samples, 1), std::runtime_error);
    UTEST_CHECK_THROW(xclass_stats_t::make_feature_stats(dataset, samples, 2), std::runtime_error);

    for (const auto batch : {1, 2, 3, 10})
    {
        const auto [st0, st1, st2, st3] = std::make_tuple(1.5811388300842, 15.811388300842, 1.0, 0.44721359549996);

        check_xclass_stats(xclass_stats_t::make_feature_stats(dataset, samples, 0), mclass_hashes,
                           make_indices(2, 2, 1, 1), make_indices(1, 0, 0, 1, 2, 3),
                           make_scalars(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0));

        check_scalar_stats(scalar_stats_t::make_feature_stats(dataset, samples, 1, batch), make_indices(6),
                           make_scalars(1.0), make_scalars(1.0), make_scalars(1.0), make_scalars(0.0),
                           make_scalars(1e+8), make_scalars(1e-8), make_scalars(1e+8), make_scalars(1e-8));

        check_scalar_stats(scalar_stats_t::make_feature_stats(dataset, samples, 2, batch), make_indices(5, 5, 5, 5),
                           make_scalars(1.0, 10.0, -3.0, 0.0), make_scalars(5.0, 50.0, -1.0, 1.0),
                           make_scalars(3.0, 30.0, -2.0, 0.2), make_scalars(st0, st1, st2, st3),
                           make_scalars(1.0 / 4.0, 1.0 / 40.0, 1.0 / 2.0, 1.0 / 1.0), make_scalars(4.0, 40.0, 2.0, 1.0),
                           make_scalars(1.0 / st0, 1.0 / st1, 1.0 / st2, 1.0 / st3), make_scalars(st0, st1, st2, st3));

        check_scalar_stats(scalar_stats_t::make_flatten_stats(dataset, samples, batch),
                           make_indices(6, 6, 6, 6, 5, 5, 5, 5), make_scalars(0.0, 0.0, 0.0, 1.0, 1.0, 10.0, -3.0, 0.0),
                           make_scalars(0.0, 0.0, 0.0, 1.0, 5.0, 50.0, -1.0, 1.0),
                           make_scalars(0.0, 0.0, 0.0, 1.0, 3.0, 30.0, -2.0, 0.2),
                           make_scalars(0.0, 0.0, 0.0, 0.0, st0, st1, st2, st3),
                           make_scalars(1.0, 1.0, 1.0, 1e+8, 1.0 / 4.0, 1.0 / 40.0, 1.0 / 2.0, 1.0 / 1.0),
                           make_scalars(1.0, 1.0, 1.0, 1e-8, 4.0, 40.0, 2.0, 1.0),
                           make_scalars(1.0, 1.0, 1.0, 1e+8, 1.0 / st0, 1.0 / st1, 1.0 / st2, 1.0 / st3),
                           make_scalars(1.0, 1.0, 1.0, 1e-8, st0, st1, st2, st3));

        check_xclass_stats(xclass_stats_t::make_targets_stats(dataset, samples), sclass_hashes, make_indices(4, 2),
                           make_indices(0, 1, 0, 0, 0, 1),
                           make_scalars(1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0));

        check_scalar_stats(scalar_stats_t::make_targets_stats(dataset, samples, batch), make_indices(6, 6),
                           make_scalars(0.0, 0.0), make_scalars(0.0, 0.0), make_scalars(0.0, 0.0),
                           make_scalars(0.0, 0.0), make_scalars(1.0, 1.0), make_scalars(1.0, 1.0),
                           make_scalars(1.0, 1.0), make_scalars(1.0, 1.0));
    }
}

UTEST_CASE(target_mclass)
{
    const auto datasource = make_stats_datasource(1U);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    const auto sclass_hashes = make_hashes(datasource.sclass_data());
    const auto mclass_hashes = make_hashes(datasource.mclass_data());

    UTEST_CHECK_THROW(scalar_stats_t::make_feature_stats(dataset, samples, 0), std::runtime_error);
    UTEST_CHECK_THROW(xclass_stats_t::make_feature_stats(dataset, samples, 1), std::runtime_error);
    UTEST_CHECK_THROW(xclass_stats_t::make_feature_stats(dataset, samples, 2), std::runtime_error);

    for (const auto batch : {1, 2, 3, 10})
    {
        const auto [st0, st1, st2, st3] = std::make_tuple(1.5811388300842, 15.811388300842, 1.0, 0.44721359549996);

        check_xclass_stats(xclass_stats_t::make_feature_stats(dataset, samples, 0), sclass_hashes, make_indices(3, 2),
                           make_indices(0, 1, -1, 0, 0, 1), make_scalars(0.4, 0.6, 0.0, 0.4, 0.4, 0.6));

        check_scalar_stats(scalar_stats_t::make_feature_stats(dataset, samples, 1, batch), make_indices(6),
                           make_scalars(1.0), make_scalars(1.0), make_scalars(1.0), make_scalars(0.0),
                           make_scalars(1e+8), make_scalars(1e-8), make_scalars(1e+8), make_scalars(1e-8));

        check_scalar_stats(scalar_stats_t::make_feature_stats(dataset, samples, 2, batch), make_indices(5, 5, 5, 5),
                           make_scalars(1.0, 10.0, -3.0, 0.0), make_scalars(5.0, 50.0, -1.0, 1.0),
                           make_scalars(3.0, 30.0, -2.0, 0.2), make_scalars(st0, st1, st2, st3),
                           make_scalars(1.0 / 4.0, 1.0 / 40.0, 1.0 / 2.0, 1.0 / 1.0), make_scalars(4.0, 40.0, 2.0, 1.0),
                           make_scalars(1.0 / st0, 1.0 / st1, 1.0 / st2, 1.0 / st3), make_scalars(st0, st1, st2, st3));

        check_scalar_stats(scalar_stats_t::make_flatten_stats(dataset, samples, batch), make_indices(5, 6, 5, 5, 5, 5),
                           make_scalars(0.0, 1.0, 1.0, 10.0, -3.0, 0.0), make_scalars(0.0, 1.0, 5.0, 50.0, -1.0, 1.0),
                           make_scalars(0.0, 1.0, 3.0, 30.0, -2.0, 0.2), make_scalars(0.0, 0.0, st0, st1, st2, st3),
                           make_scalars(1.0, 1e+8, 1.0 / 4.0, 1.0 / 40.0, 1.0 / 2.0, 1.0 / 1.0),
                           make_scalars(1.0, 1e-8, 4.0, 40.0, 2.0, 1.0),
                           make_scalars(1.0, 1e+8, 1.0 / st0, 1.0 / st1, 1.0 / st2, 1.0 / st3),
                           make_scalars(1.0, 1e-8, st0, st1, st2, st3));

        check_xclass_stats(xclass_stats_t::make_targets_stats(dataset, samples), mclass_hashes,
                           make_indices(2, 2, 1, 1), make_indices(1, 0, 0, 1, 2, 3),
                           make_scalars(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0));

        check_scalar_stats(scalar_stats_t::make_targets_stats(dataset, samples, batch), make_indices(6, 6, 6),
                           make_scalars(0.0, 0.0, 0.0), make_scalars(0.0, 0.0, 0.0), make_scalars(0.0, 0.0, 0.0),
                           make_scalars(0.0, 0.0, 0.0), make_scalars(1.0, 1.0, 1.0), make_scalars(1.0, 1.0, 1.0),
                           make_scalars(1.0, 1.0, 1.0), make_scalars(1.0, 1.0, 1.0));
    }
}

UTEST_CASE(target_struct)
{
    const auto datasource = make_stats_datasource(3U);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());

    const auto sclass_hashes = make_hashes(datasource.sclass_data());
    const auto mclass_hashes = make_hashes(datasource.mclass_data());

    UTEST_CHECK_THROW(xclass_stats_t::make_targets_stats(dataset, samples), std::runtime_error);

    UTEST_CHECK_THROW(scalar_stats_t::make_feature_stats(dataset, samples, 0), std::runtime_error);
    UTEST_CHECK_THROW(scalar_stats_t::make_feature_stats(dataset, samples, 1), std::runtime_error);
    UTEST_CHECK_THROW(xclass_stats_t::make_feature_stats(dataset, samples, 2), std::runtime_error);

    for (const auto batch : {1, 2, 3, 10})
    {
        const auto [st0, st1, st2, st3] =
            std::make_tuple(1.870828693387, 18.708286933870, 1.211060141639, 0.408248290464);

        check_xclass_stats(xclass_stats_t::make_feature_stats(dataset, samples, 0), sclass_hashes, make_indices(3, 2),
                           make_indices(0, 1, -1, 0, 0, 1), make_scalars(0.4, 0.6, 0.0, 0.4, 0.4, 0.6));

        check_xclass_stats(xclass_stats_t::make_feature_stats(dataset, samples, 1), mclass_hashes,
                           make_indices(2, 2, 1, 1), make_indices(1, 0, 0, 1, 2, 3),
                           make_scalars(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0));

        check_scalar_stats(scalar_stats_t::make_feature_stats(dataset, samples, 2, batch), make_indices(6),
                           make_scalars(1.0), make_scalars(1.0), make_scalars(1.0), make_scalars(0.0),
                           make_scalars(1e+8), make_scalars(1e-8), make_scalars(1e+8), make_scalars(1e-8));

        check_scalar_stats(scalar_stats_t::make_flatten_stats(dataset, samples, batch), make_indices(5, 6, 6, 6, 6),
                           make_scalars(0.0, 0.0, 0.0, 0.0, 1.0), make_scalars(0.0, 0.0, 0.0, 0.0, 1.0),
                           make_scalars(0.0, 0.0, 0.0, 0.0, 1.0), make_scalars(0.0, 0.0, 0.0, 0.0, 0.0),
                           make_scalars(1.0, 1.0, 1.0, 1.0, 1e+8), make_scalars(1.0, 1.0, 1.0, 1.0, 1e-8),
                           make_scalars(1.0, 1.0, 1.0, 1.0, 1e+8), make_scalars(1.0, 1.0, 1.0, 1.0, 1e-8));

        check_scalar_stats(scalar_stats_t::make_targets_stats(dataset, samples, batch), make_indices(6, 6, 6, 6),
                           make_scalars(0.0, 0.0, -3.0, 0.0), make_scalars(5.0, 50.0, 0.0, 1.0),
                           make_scalars(2.5, 25.0, -10.0 / 6.0, 1.0 / 6.0), make_scalars(st0, st1, st2, st3),
                           make_scalars(1.0 / 5.0, 1.0 / 50.0, 1.0 / 3.0, 1.0 / 1.0), make_scalars(5.0, 50.0, 3.0, 1.0),
                           make_scalars(1.0 / st0, 1.0 / st1, 1.0 / st2, 1.0 / st3), make_scalars(st0, st1, st2, st3));
    }
}

UTEST_END_MODULE()
