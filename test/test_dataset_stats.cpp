#include <nano/dataset/stats.h>
#include <utest/utest.h>

using namespace nano;

constexpr auto INF = std::numeric_limits<scalar_t>::infinity();
constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

template <typename tscalar, size_t trank>
static void check_sclass_stats(const feature_t& feature, dataset_iterator_t<tscalar, trank> it,
                               tensor_size_t expected_samples, const indices_t& expected_class_counts,
                               tensor1d_t expected_weights, scalar_t epsilon = 1e-12)
{
    const auto stats = sclass_stats_t::make(feature, it);

    UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
    UTEST_CHECK_EQUAL(stats.class_counts(), expected_class_counts);
    {
        if (expected_samples > 0)
        {
            expected_weights.array() *= static_cast<scalar_t>(expected_samples) / expected_weights.sum();
        }
        const auto weights = stats.sample_weights(feature, it);

        UTEST_CHECK_EQUAL(weights.size(), it.size());
        UTEST_CHECK_CLOSE(weights.sum(), expected_samples, epsilon);
        UTEST_CHECK_CLOSE(weights, expected_weights, epsilon);
    }
    {
        // sample weights for incompatible features
        const auto weights0          = stats.sample_weights(feature_t{""}.sclass(42), it);
        const auto expected_weights0 = make_full_tensor<scalar_t>(make_dims(it.size()), 0.0);
        UTEST_CHECK_CLOSE(weights0, expected_weights0, epsilon);
    }
}

template <typename tscalar, size_t trank>
static void check_mclass_stats(const feature_t& feature, dataset_iterator_t<tscalar, trank> it,
                               tensor_size_t expected_samples, const indices_t& expected_class_counts,
                               tensor1d_t expected_weights, scalar_t epsilon = 1e-12)
{
    const auto stats = mclass_stats_t::make(feature, it);

    UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
    UTEST_CHECK_EQUAL(stats.class_counts(), expected_class_counts);
    {
        if (expected_samples > 0)
        {
            expected_weights.array() *= static_cast<scalar_t>(expected_samples) / expected_weights.sum();
        }
        const auto weights = stats.sample_weights(feature, it);

        UTEST_CHECK_EQUAL(weights.size(), it.size());
        UTEST_CHECK_CLOSE(weights.sum(), expected_samples, epsilon);
        UTEST_CHECK_CLOSE(weights, expected_weights, epsilon);
    }
    {
        // sample weights for incompatible features
        const auto weights0          = stats.sample_weights(feature_t{""}.sclass(42), it);
        const auto expected_weights0 = make_full_tensor<scalar_t>(make_dims(it.size()), 0.0);
        UTEST_CHECK_CLOSE(weights0, expected_weights0, epsilon);
    }
}

template <typename tscalar, size_t trank>
static auto check_scalar_stats(const feature_t& feature, dataset_iterator_t<tscalar, trank> it,
                               tensor_size_t expected_samples, scalar_t expected_min, scalar_t expected_max,
                               scalar_t expected_mean, scalar_t expected_stdev, scalar_t epsilon = 1e-12)
{
    auto stats = scalar_stats_t::make(feature, it);

    const auto gt_min     = make_full_tensor<scalar_t>(feature.dims(), expected_min);
    const auto gt_max     = make_full_tensor<scalar_t>(feature.dims(), expected_max);
    const auto gt_mean    = make_full_tensor<scalar_t>(feature.dims(), expected_mean);
    const auto gt_stdev   = make_full_tensor<scalar_t>(feature.dims(), expected_stdev);
    const auto gt_samples = make_full_tensor<tensor_size_t>(make_dims(gt_min.size()), expected_samples);

    UTEST_CHECK_EQUAL(stats.samples(), gt_samples);
    UTEST_CHECK_CLOSE(stats.min(), gt_min.reshape(-1), epsilon);
    UTEST_CHECK_CLOSE(stats.max(), gt_max.reshape(-1), epsilon);
    UTEST_CHECK_CLOSE(stats.mean(), gt_mean.reshape(-1), epsilon);
    UTEST_CHECK_CLOSE(stats.stdev(), gt_stdev.reshape(-1), epsilon);

    return stats;
}

static void check_scaling(const scalar_stats_t& stats, scaling_type scaling, tensor4d_t values,
                          const tensor4d_t& expected_scaled_values, const tensor4d_t& expected_upscaled_values,
                          scalar_t epsilon = 1e-12)
{
    UTEST_CHECK_NOTHROW(stats.scale(scaling, values.tensor()));
    UTEST_CHECK_CLOSE(values, expected_scaled_values, epsilon);

    UTEST_CHECK_NOTHROW(stats.upscale(scaling, values.tensor()));
    UTEST_CHECK_CLOSE(values, expected_upscaled_values, epsilon);

    const auto scalings = {
        scaling_type::none,
        scaling_type::mean,
        scaling_type::minmax,
        scaling_type::standard,
    };

    // check upscaling of affine transformations
    for (const auto flatten_scaling : scalings)
    {
        for (const auto targets_scaling : scalings)
        {
            // simulate a linear model ...
            const auto fsize  = stats.size();
            const auto tsize  = 13;
            const auto trials = 100;

            auto flatten = make_random_tensor<scalar_t>(make_dims(trials, fsize));
            auto targets = make_random_tensor<scalar_t>(make_dims(trials, tsize));
            auto weights = make_random_tensor<scalar_t>(make_dims(tsize, fsize));
            auto bias    = make_random_tensor<scalar_t>(make_dims(tsize));

            targets.matrix() = flatten.matrix() * weights.matrix().transpose();
            targets.matrix().rowwise() += bias.vector().transpose();

            auto flatten_stats = scalar_stats_t{fsize};
            auto targets_stats = scalar_stats_t{tsize};
            for (tensor_size_t i = 0; i < trials; ++i)
            {
                flatten_stats += flatten.array(i);
                targets_stats += targets.array(i);
            }
            flatten_stats.done();
            targets_stats.done();

            flatten_stats.upscale(flatten_scaling, flatten);
            targets_stats.upscale(targets_scaling, targets);
            ::nano::upscale(flatten_stats, flatten_scaling, targets_stats, targets_scaling, weights, bias);

            for (tensor_size_t i = 0; i < trials; ++i)
            {
                UTEST_CHECK_CLOSE(weights.matrix() * flatten.vector(i) + bias.vector(), targets.vector(i), epsilon);
            }
        }
    }
}

UTEST_BEGIN_MODULE(test_dataset_stats)

UTEST_CASE(scalar)
{
    for (const auto dims : {make_dims(3, 1, 2), make_dims(1, 1, 1)})
    {
        const auto make_values = [&](scalar_t value0, scalar_t value1, scalar_t value2)
        {
            tensor4d_t values(cat_dims(3, dims));
            values.tensor(0).full(value0);
            values.tensor(1).full(value1);
            values.tensor(2).full(value2);
            return values;
        };

        const auto samples = arange(0, 42);
        const auto feature = feature_t{"feature"}.scalar(feature_type::float32, dims);

        auto       mask = make_mask(make_dims(samples.size()));
        auto       data = make_full_tensor<scalar_t>(cat_dims(samples.size(), dims), NaN);
        const auto it   = make_iterator(data, mask, samples);
        {
            check_scalar_stats(feature, it, 0, 0.0, 0.0, 0.0, 0.0);
        }
        {
            data.tensor(0).full(1.0);
            setbit(mask, 0);
            const auto stats = check_scalar_stats(feature, it, 1, 1.0, 1.0, 1.0, 0.0);

            const auto values = make_values(0.50, NaN, 0.75);

            auto tmp_values = values;
            UTEST_CHECK_THROW(stats.scale(static_cast<scaling_type>(-1), tmp_values.tensor()), std::runtime_error);
            UTEST_CHECK_THROW(stats.upscale(static_cast<scaling_type>(-1), tmp_values.tensor()), std::runtime_error);

            check_scaling(stats, scaling_type::none, values, make_values(0.50, 0.00, 0.75),
                          make_values(0.50, 0.00, 0.75));

            check_scaling(stats, scaling_type::mean, values, make_values(-0.50, 0.00, -0.25),
                          make_values(0.50, 1.00, 0.75));

            check_scaling(stats, scaling_type::minmax, values, make_values(-0.50, 0.00, -0.25),
                          make_values(0.50, 1.00, 0.75));

            check_scaling(stats, scaling_type::standard, values, make_values(-0.50, 0.00, -0.25),
                          make_values(0.50, 1.00, 0.75));
        }
        {
            for (tensor_size_t sample = 1; sample < samples.size(); sample += 3)
            {
                data.tensor(sample).full(static_cast<scalar_t>(sample));
                setbit(mask, sample);
            }
            const auto stats = check_scalar_stats(feature, it, 15, 1.0, 40.0, 19.2, 13.09961831505);

            const auto values = make_values(1.0, NaN, 7.0);

            check_scaling(stats, scaling_type::none, values, make_values(1.0, 0.0, 7.0), make_values(1.0, 0.0, 7.0));

            check_scaling(stats, scaling_type::mean, values, make_values(-18.2 / 39.0, 0.0, -12.2 / 39.0),
                          make_values(1.0, 19.2, 7.0));

            check_scaling(stats, scaling_type::minmax, values, make_values(0.0, 0.0, 6.0 / 39.0),
                          make_values(1.0, 1.0, 7.0));

            check_scaling(stats, scaling_type::standard, values,
                          make_values(-18.2 / 13.09961831505, 0.0, -12.2 / 13.09961831505),
                          make_values(1.0, 19.2, 7.0));
        }
    }
}

UTEST_CASE(sclass)
{
    const auto samples = arange(0, 20);
    const auto feature = feature_t{"feature"}.sclass(3);

    auto       mask = make_mask(make_dims(samples.size()));
    auto       data = make_full_tensor<uint8_t>(make_dims(samples.size()), 0x00);
    const auto it   = make_iterator(data, mask, samples);
    {
        const auto stats = sclass_stats_t{};
        UTEST_CHECK_EQUAL(stats.samples(), 0);
    }
    {
        check_sclass_stats(feature, it, 0, make_indices(0, 0, 0),
                           make_tensor<scalar_t>(make_dims(20), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
    {
        data(0) = static_cast<uint8_t>(0);
        setbit(mask, 0);

        check_sclass_stats(feature, it, 1, make_indices(1, 0, 0),
                           make_tensor<scalar_t>(make_dims(20), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
    {
        data(1) = static_cast<uint8_t>(1);
        setbit(mask, 1);
        data(3) = static_cast<uint8_t>(2);
        setbit(mask, 3);
        data(5) = static_cast<uint8_t>(0);
        setbit(mask, 5);
        data(6) = static_cast<uint8_t>(1);
        setbit(mask, 6);
        data(9) = static_cast<uint8_t>(1);
        setbit(mask, 9);

        check_sclass_stats(feature, it, 6, make_indices(2, 3, 1),
                           make_tensor<scalar_t>(make_dims(20), 1.0 / 2.0, 1.0 / 3.0, 0.0, 1.0, 0.0, 1.0 / 2.0,
                                                 1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0));
    }
    {
        data(10) = static_cast<uint8_t>(2);
        setbit(mask, 10);
        data(11) = static_cast<uint8_t>(2);
        setbit(mask, 11);
        data(13) = static_cast<uint8_t>(2);
        setbit(mask, 13);
        data(15) = static_cast<uint8_t>(0);
        setbit(mask, 15);
        data(16) = static_cast<uint8_t>(1);
        setbit(mask, 16);
        data(19) = static_cast<uint8_t>(1);
        setbit(mask, 19);

        check_sclass_stats(feature, it, 12, make_indices(3, 5, 4),
                           make_tensor<scalar_t>(make_dims(20), 1.0 / 3.0, 1.0 / 5.0, 0.0, 1.0 / 4.0, 0.0, 1.0 / 3.0,
                                                 1.0 / 5.0, 0.0, 0.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 4.0, 0.0, 1.0 / 4.0,
                                                 0.0, 1.0 / 3.0, 1.0 / 5.0, 0.0, 0.0, 1.0 / 5.0));
    }
}

UTEST_CASE(mclass)
{
    const auto samples = arange(0, 22);
    const auto feature = feature_t{"feature"}.sclass(3);

    auto       mask = make_mask(make_dims(samples.size()));
    auto       data = make_full_tensor<uint8_t>(make_dims(samples.size(), feature.classes()), 0x00);
    const auto it   = make_iterator(data, mask, samples);
    {
        const auto stats = mclass_stats_t{};
        UTEST_CHECK_EQUAL(stats.samples(), 0);
    }
    {
        check_mclass_stats(feature, it, 0, make_indices(0, 0, 0, 0, 0, 0),
                           make_tensor<scalar_t>(make_dims(22), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
    {
        data.tensor(3) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1);
        setbit(mask, 3);
        data.tensor(5) = make_tensor<uint8_t>(make_dims(3), 1, 1, 1);
        setbit(mask, 5);
        data.tensor(8) = make_tensor<uint8_t>(make_dims(3), 0, 0, 1);
        setbit(mask, 8);

        check_mclass_stats(feature, it, 3, make_indices(0, 0, 0, 1, 1, 1),
                           make_tensor<scalar_t>(make_dims(22), 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
    {
        data.tensor(11) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1);
        setbit(mask, 11);
        data.tensor(12) = make_tensor<uint8_t>(make_dims(3), 1, 1, 1);
        setbit(mask, 12);
        data.tensor(13) = make_tensor<uint8_t>(make_dims(3), 1, 0, 1);
        setbit(mask, 13);
        data.tensor(14) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1);
        setbit(mask, 14);

        check_mclass_stats(feature, it, 7, make_indices(0, 0, 0, 1, 4, 2),
                           make_tensor<scalar_t>(make_dims(22), 0.0, 0.0, 0.0, 1.0 / 4.0, 0.0, 1.0 / 2.0, 0.0, 0.0, 1.0,
                                                 0.0, 0.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0));
    }
    {
        data.tensor(15) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0);
        setbit(mask, 15);
        data.tensor(16) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0);
        setbit(mask, 16);
        data.tensor(17) = make_tensor<uint8_t>(make_dims(3), 0, 0, 1);
        setbit(mask, 17);
        data.tensor(18) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1);
        setbit(mask, 18);
        data.tensor(19) = make_tensor<uint8_t>(make_dims(3), 0, 1, 1);
        setbit(mask, 19);
        data.tensor(20) = make_tensor<uint8_t>(make_dims(3), 0, 0, 0);
        setbit(mask, 20);
        data.tensor(21) = make_tensor<uint8_t>(make_dims(3), 0, 1, 0);
        setbit(mask, 21);

        check_mclass_stats(feature, it, 14, make_indices(3, 0, 1, 2, 6, 2),
                           make_tensor<scalar_t>(make_dims(22), 0.0, 0.0, 0.0, 1.0 / 6.0, 0.0, 1.0 / 2.0, 0.0, 0.0,
                                                 1.0 / 2.0, 0.0, 0.0, 1.0 / 6.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 6.0,
                                                 1.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0,
                                                 1.0));
    }
}

UTEST_CASE(flatten)
{
    auto flatten = tensor1d_t{4};
    auto stats   = scalar_stats_t{flatten.size()};

    flatten(0) = 1.0;
    flatten(1) = NaN;
    flatten(2) = 5.0;
    flatten(3) = NaN;
    stats += flatten.array();
    flatten(0) = 2.0;
    flatten(1) = 0.0;
    flatten(2) = 5.0;
    flatten(3) = INF;
    stats += flatten.array();
    flatten(0) = NaN;
    flatten(1) = 5.0;
    flatten(2) = 6.0;
    flatten(3) = NaN;
    stats += flatten.array();
    flatten(0) = 4.0;
    flatten(1) = 3.0;
    flatten(2) = 5.0;
    flatten(3) = NaN;
    stats += flatten.array();
    stats.done();

    UTEST_CHECK_EQUAL(stats.samples(), make_tensor<tensor_size_t>(make_dims(4), 3, 3, 4, 0));
    UTEST_CHECK_CLOSE(stats.min(), make_tensor<scalar_t>(make_dims(4), 1.0, 0.0, 5.0, 0.0), 1e-12);
    UTEST_CHECK_CLOSE(stats.max(), make_tensor<scalar_t>(make_dims(4), 4.0, 5.0, 6.0, 0.0), 1e-12);
    UTEST_CHECK_CLOSE(stats.mean(), make_tensor<scalar_t>(make_dims(4), 7.0 / 3.0, 8.0 / 3.0, 21.0 / 4.0, 0.0), 1e-12);
    UTEST_CHECK_CLOSE(stats.stdev(), make_tensor<scalar_t>(make_dims(4), 1.527525231652, 2.516611478424, 0.5, 0.0),
                      1e-12);

    const auto dims   = make_dims(1, 4, 1, 1);
    const auto values = make_full_tensor<scalar_t>(dims, 1.0);

    check_scaling(stats, scaling_type::none, values, make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0),
                  make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0));

    check_scaling(stats, scaling_type::mean, values,
                  make_tensor<scalar_t>(dims, -4.0 / 9.0, -1.0 / 3.0, -17.0 / 4.0, 1.0),
                  make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0));

    check_scaling(stats, scaling_type::minmax, values, make_tensor<scalar_t>(dims, 0.0, 1.0 / 5.0, -4.0 / 1.0, 1.0),
                  make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0));

    check_scaling(
        stats, scaling_type::standard, values,
        make_tensor<scalar_t>(dims, -4.0 / 3.0 / 1.527525231652, -5.0 / 3.0 / 2.516611478424, -17.0 / 4.0 / 0.5, 1.0),
        make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0));
}

UTEST_CASE(flatten_enable_scaling)
{
    auto flatten = tensor1d_t{4};
    auto stats   = scalar_stats_t{flatten.size()};

    const auto enable_scaling = make_tensor<uint8_t>(make_dims(4), 0x00, 0x01, 0x01, 0x00);

    flatten(0) = 1.0;
    flatten(1) = NaN;
    flatten(2) = 5.0;
    flatten(3) = NaN;
    stats += flatten.array();
    flatten(0) = 2.0;
    flatten(1) = 0.0;
    flatten(2) = 5.0;
    flatten(3) = INF;
    stats += flatten.array();
    flatten(0) = NaN;
    flatten(1) = 5.0;
    flatten(2) = 6.0;
    flatten(3) = NaN;
    stats += flatten.array();
    flatten(0) = 4.0;
    flatten(1) = 3.0;
    flatten(2) = 5.0;
    flatten(3) = NaN;
    stats += flatten.array();
    stats.done(enable_scaling);

    UTEST_CHECK_EQUAL(stats.samples(), make_tensor<tensor_size_t>(make_dims(4), 3, 3, 4, 0));
    UTEST_CHECK_CLOSE(stats.min(), make_tensor<scalar_t>(make_dims(4), 0.0, 0.0, 5.0, 0.0), 1e-12);
    UTEST_CHECK_CLOSE(stats.max(), make_tensor<scalar_t>(make_dims(4), 0.0, 5.0, 6.0, 0.0), 1e-12);
    UTEST_CHECK_CLOSE(stats.mean(), make_tensor<scalar_t>(make_dims(4), 0.0, 8.0 / 3.0, 21.0 / 4.0, 0.0), 1e-12);
    UTEST_CHECK_CLOSE(stats.stdev(), make_tensor<scalar_t>(make_dims(4), 0.0, 2.516611478424, 0.5, 0.0), 1e-12);

    const auto dims   = make_dims(1, 4, 1, 1);
    const auto values = make_full_tensor<scalar_t>(dims, 1.0);

    check_scaling(stats, scaling_type::none, values, make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0),
                  make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0));

    check_scaling(stats, scaling_type::mean, values, make_tensor<scalar_t>(dims, 1.0, -1.0 / 3.0, -17.0 / 4.0, 1.0),
                  make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0));

    check_scaling(stats, scaling_type::minmax, values, make_tensor<scalar_t>(dims, 1.0, 1.0 / 5.0, -4.0 / 1.0, 1.0),
                  make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0));

    check_scaling(stats, scaling_type::standard, values,
                  make_tensor<scalar_t>(dims, 1.0, -5.0 / 3.0 / 2.516611478424, -17.0 / 4.0 / 0.5, 1.0),
                  make_tensor<scalar_t>(dims, 1.0, 1.0, 1.0, 1.0));
}

UTEST_END_MODULE()
