#pragma once

#include <nano/dataset.h>
#include <nano/dataset/iterator.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::parallel;

static constexpr auto N   = std::numeric_limits<scalar_t>::quiet_NaN();
static constexpr auto Na  = std::numeric_limits<scalar_t>::quiet_NaN();
static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();
static constexpr auto INF = std::numeric_limits<scalar_t>::infinity();

static auto make_sample_splits(const dataset_t& dataset)
{
    const auto samples = dataset.samples();

    return std::vector<indices_t>{arange(0, samples), arange(0, samples / 2), arange(samples / 2, samples)};
}

template <typename tgenerator>
static void add_generator(dataset_t& dataset)
{
    UTEST_REQUIRE_NOTHROW(dataset.add<tgenerator>());
}

template <typename tgenerator>
static void add_generator(dataset_t& dataset, indices_t features)
{
    UTEST_REQUIRE_NOTHROW(dataset.add<tgenerator>(std::move(features)));
}

template <typename tgenerator>
static void add_generator(dataset_t& dataset, indices_t features1, indices_t features2)
{
    UTEST_REQUIRE_NOTHROW(dataset.add<tgenerator>(std::move(features1), std::move(features2)));
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
[[maybe_unused]] static void check_select0(const select_iterator_t& iterator, indices_cmap_t samples,
                                           indices_cmap_t features, const tensor_t<tstorage, tscalar, trank>& expected)
{
    const auto& dataset = iterator.dataset();

    const auto expected_feature = features(0);
    using tvalues               = decltype(expected.tensor());

    const auto checker = [&](const auto& expected_values)
    {
        UTEST_CHECK_NOTHROW(iterator.loop(samples, features,
                                          [&](tensor_size_t feature, size_t, tvalues values)
                                          {
                                              UTEST_CHECK_EQUAL(feature, expected_feature);
                                              UTEST_CHECK_CLOSE(values, expected_values, 1e-12);
                                          }));
    };

    checker(expected.indexed(samples));

    dataset.shuffle(expected_feature);
    const auto shuffle = dataset.shuffled(expected_feature, samples);
    UTEST_REQUIRE_EQUAL(shuffle.size(), samples.size());
    checker(expected.indexed(shuffle));

    const auto shuffle2 = dataset.shuffled(expected_feature, samples);
    UTEST_CHECK_EQUAL(shuffle, shuffle2);

    dataset.unshuffle();
    checker(expected.indexed(samples));

    dataset.drop(expected_feature);
    tensor_t<tstorage, tscalar, trank> expected_dropped = expected.indexed(samples);
    switch (dataset.feature(expected_feature).type())
    {
    case feature_type::sclass: expected_dropped.full(-1); break; // NOLINT(bugprone-branch-clone)
    case feature_type::mclass: expected_dropped.full(-1); break;
    default: expected_dropped.full(static_cast<tscalar>(NaN)); break;
    }
    checker(expected_dropped);

    dataset.undrop();
    checker(expected.indexed(samples));
}

[[maybe_unused]] static void check_select(const select_iterator_t& iterator, indices_cmap_t samples,
                                          indices_cmap_t features, const sclass_mem_t& expected)
{
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, mclass_cmap_t) {}),
                      std::runtime_error);
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, scalar_cmap_t) {}),
                      std::runtime_error);
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, struct_cmap_t) {}),
                      std::runtime_error);
    check_select0(iterator, samples, features, expected);
}

[[maybe_unused]] static void check_select(const select_iterator_t& iterator, indices_cmap_t samples,
                                          indices_cmap_t features, const mclass_mem_t& expected)
{
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, sclass_cmap_t) {}),
                      std::runtime_error);
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, scalar_cmap_t) {}),
                      std::runtime_error);
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, struct_cmap_t) {}),
                      std::runtime_error);
    check_select0(iterator, samples, features, expected);
}

[[maybe_unused]] static void check_select(const select_iterator_t& iterator, indices_cmap_t samples,
                                          indices_cmap_t features, const scalar_mem_t& expected)
{
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, sclass_cmap_t) {}),
                      std::runtime_error);
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, mclass_cmap_t) {}),
                      std::runtime_error);
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, struct_cmap_t) {}),
                      std::runtime_error);
    check_select0(iterator, samples, features, expected);
}

[[maybe_unused]] static void check_select(const select_iterator_t& iterator, indices_cmap_t samples,
                                          indices_cmap_t features, const struct_mem_t& expected)
{
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, sclass_cmap_t) {}),
                      std::runtime_error);
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, mclass_cmap_t) {}),
                      std::runtime_error);
    UTEST_CHECK_THROW(iterator.loop(samples, features, [&](tensor_size_t, size_t, scalar_cmap_t) {}),
                      std::runtime_error);
    check_select0(iterator, samples, features, expected);
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
[[maybe_unused]] static void check_select(const dataset_t& dataset, const tensor_size_t feature,
                                          const tensor_t<tstorage, tscalar, trank>& expected)
{
    auto iterator = select_iterator_t{dataset};

    const auto features = make_indices(feature);
    for (const auto& samples : make_sample_splits(dataset))
    {
        check_select(iterator, samples, features, expected);
    }
}

[[maybe_unused]] static void check_flatten(const dataset_t& dataset, const tensor2d_t& expected_flatten,
                                           const indices_t& expected_column2features, const bool dropped = false,
                                           const scalar_t eps = 1e-12)
{
    UTEST_REQUIRE_EQUAL(dataset.columns(), expected_flatten.size<1>());
    UTEST_REQUIRE_EQUAL(dataset.columns(), expected_column2features.size());

    for (tensor_size_t column = 0; column < dataset.columns(); ++column)
    {
        UTEST_CHECK_EQUAL(dataset.column2feature(column), expected_column2features(column));
    }

    for (const auto& samples : make_sample_splits(dataset))
    {
        auto iterator = flatten_iterator_t{dataset, samples};

        for (const auto batch : {2, 3, 8})
        {
            iterator.batch(batch);
            for (const auto scaling : enum_values<scaling_type>())
            {
                iterator.scaling(scaling);
                UTEST_CHECK_EQUAL(iterator.scaling(), scaling);

                if (batch == 2)
                {
                    UTEST_CHECK(!iterator.cache_flatten(0U));
                }
                else
                {
                    UTEST_CHECK(iterator.cache_flatten(1U << 24));
                }

                const auto& stats                   = iterator.flatten_stats();
                auto        expected_scaled_flatten = expected_flatten;
                stats.scale(scaling, expected_scaled_flatten);

                {
                    auto called = make_full_tensor<tensor_size_t>(make_dims(samples.size()), 0);
                    UTEST_CHECK_NOTHROW(iterator.loop(
                        [&](const tensor_range_t range, const size_t tnum, tensor2d_cmap_t flatten)
                        {
                            called.slice(range).full(1);
                            UTEST_CHECK_GREATER_EQUAL(tnum, 0U);
                            UTEST_CHECK_LESS(tnum, pool_t::max_size());
                            UTEST_CHECK_LESS_EQUAL(range.size(), batch);
                            UTEST_CHECK_GREATER_EQUAL(range.begin(), 0);
                            UTEST_CHECK_LESS_EQUAL(range.end(), samples.size());
                            UTEST_REQUIRE_CLOSE(flatten, expected_scaled_flatten.indexed(samples.slice(range)), eps);
                        }));
                    UTEST_CHECK_EQUAL(called, make_full_tensor<tensor_size_t>(make_dims(samples.size()), 1));
                }
                if (!dropped)
                {
                    // NB: also test with shuffling the columns associated to the first feature
                    // NB: caching needs to be disabled (to make sure the old values are not reused)
                    iterator.cache_flatten(0U);

                    const auto feature_to_shuffle = 0;
                    dataset.shuffle(feature_to_shuffle);
                    const auto shuffle = dataset.shuffled(feature_to_shuffle, samples);
                    UTEST_REQUIRE_EQUAL(shuffle.size(), samples.size());

                    auto called = make_full_tensor<tensor_size_t>(make_dims(samples.size()), 0);
                    UTEST_CHECK_NOTHROW(iterator.loop(
                        [&](const tensor_range_t range, const size_t tnum, tensor2d_cmap_t flatten)
                        {
                            called.slice(range).full(1);
                            UTEST_CHECK_GREATER_EQUAL(tnum, 0U);
                            UTEST_CHECK_LESS(tnum, pool_t::max_size());
                            UTEST_CHECK_GREATER_EQUAL(range.begin(), 0);
                            UTEST_CHECK_LESS_EQUAL(range.end(), samples.size());
                            for (tensor_size_t column = 0, columns = dataset.columns(); column < columns; ++column)
                            {
                                const auto& expected_samples =
                                    dataset.column2feature(column) == feature_to_shuffle ? shuffle : samples;

                                for (tensor_size_t index = range.begin(); index < range.end(); ++index)
                                {
                                    UTEST_REQUIRE_CLOSE(flatten(index - range.begin(), column),
                                                        expected_scaled_flatten(expected_samples(index), column), eps);
                                }
                            }
                        }));
                    UTEST_CHECK_EQUAL(called, make_full_tensor<tensor_size_t>(make_dims(samples.size()), 1));

                    dataset.unshuffle();
                }
                if (!dropped)
                {
                    // NB: test dropping all features
                    for (tensor_size_t feature = 0; feature < dataset.features(); ++feature)
                    {
                        dataset.drop(feature);
                    }

                    UTEST_CHECK_NOTHROW(iterator.loop(
                        [&](const tensor_range_t, const size_t, tensor2d_cmap_t flatten)
                        { UTEST_REQUIRE_CLOSE(flatten, make_full_tensor<scalar_t>(flatten.dims(), 0.0), eps); }));

                    dataset.undrop();
                }
            }
        }
    }
}

[[maybe_unused]] static void check_select_stats(const dataset_t& dataset, const indices_t& expected_sclass_features,
                                                const indices_t& expected_mclass_features,
                                                const indices_t& expected_scalar_features,
                                                const indices_t& expected_struct_features)
{
    UTEST_CHECK_EQUAL(make_sclass_features(dataset), expected_sclass_features);
    UTEST_CHECK_EQUAL(make_mclass_features(dataset), expected_mclass_features);
    UTEST_CHECK_EQUAL(make_scalar_features(dataset), expected_scalar_features);
    UTEST_CHECK_EQUAL(make_struct_features(dataset), expected_struct_features);

    const auto samples = arange(0, dataset.samples());

    auto features = indices_t{dataset.features()};
    auto iterator = select_iterator_t{dataset};

    const auto op_sclass = [&](tensor_size_t feature, size_t, sclass_cmap_t) { features(feature) = +1; };
    const auto op_mclass = [&](tensor_size_t feature, size_t, mclass_cmap_t) { features(feature) = +1; };
    const auto op_scalar = [&](tensor_size_t feature, size_t, scalar_cmap_t) { features(feature) = +1; };
    const auto op_struct = [&](tensor_size_t feature, size_t, struct_cmap_t) { features(feature) = +1; };

    const auto make_features = [&]()
    {
        indices_t indices(features.sum());
        for (tensor_size_t i = 0, k = 0; i < features.size(); ++i)
        {
            if (features(i) == +1)
            {
                indices(k) = i;
                ++k;
            }
        }
        return indices;
    };

    features.array() = 0;
    UTEST_CHECK_NOTHROW(iterator.loop(samples, op_sclass));
    UTEST_CHECK_EQUAL(expected_sclass_features, make_features());

    features.array() = 0;
    UTEST_CHECK_NOTHROW(iterator.loop(samples, op_mclass));
    UTEST_CHECK_EQUAL(expected_mclass_features, make_features());

    features.array() = 0;
    UTEST_CHECK_NOTHROW(iterator.loop(samples, op_scalar));
    UTEST_CHECK_EQUAL(expected_scalar_features, make_features());

    features.array() = 0;
    UTEST_CHECK_NOTHROW(iterator.loop(samples, op_struct));
    UTEST_CHECK_EQUAL(expected_struct_features, make_features());
}

[[maybe_unused]] static void check_flatten_stats0(const dataset_t& dataset, const indices_t& expected_samples,
                                                  const tensor1d_t& expected_min, const tensor1d_t& expected_max,
                                                  const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev,
                                                  const scalar_t epsilon = 1e-12)
{
    const auto samples = arange(0, dataset.samples());

    auto iterator = flatten_iterator_t{dataset, samples};
    for (const auto scaling : enum_values<scaling_type>())
    {
        iterator.batch(3);
        iterator.scaling(scaling);
        UTEST_CHECK_EQUAL(iterator.scaling(), scaling);

        const auto& stats = iterator.flatten_stats();
        UTEST_CHECK_EQUAL(stats.m_samples, expected_samples);
        UTEST_CHECK_CLOSE(stats.m_min, expected_min, epsilon);
        UTEST_CHECK_CLOSE(stats.m_max, expected_max, epsilon);
        UTEST_CHECK_CLOSE(stats.m_mean, expected_mean, epsilon);
        UTEST_CHECK_CLOSE(stats.m_stdev, expected_stdev, epsilon);
    }
}

[[maybe_unused]] static void check_flatten_stats(const dataset_t& dataset, const indices_t& expected_samples,
                                                 const tensor1d_t& expected_min, const tensor1d_t& expected_max,
                                                 const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev)
{
    check_flatten_stats0(dataset, expected_samples, expected_min, expected_max, expected_mean, expected_stdev);

    dataset.shuffle(1);
    check_flatten_stats0(dataset, expected_samples, expected_min, expected_max, expected_mean, expected_stdev);

    dataset.shuffle(0);
    check_flatten_stats0(dataset, expected_samples, expected_min, expected_max, expected_mean, expected_stdev);

    dataset.unshuffle();
    check_flatten_stats0(dataset, expected_samples, expected_min, expected_max, expected_mean, expected_stdev);
}

[[maybe_unused]] static void check_targets(const dataset_t& dataset, const feature_t& expected_target,
                                           tensor3d_dims_t expected_target_dims, const tensor4d_t& expected_targets,
                                           const scalar_t epsilon = 1e-12)
{
    UTEST_CHECK_EQUAL(dataset.target(), expected_target);
    UTEST_CHECK_EQUAL(dataset.target_dims(), expected_target_dims);

    const auto samples = arange(0, expected_targets.size<0>());

    auto iterator = targets_iterator_t{dataset, samples};
    for (const auto batch : {1, 3, 8})
    {
        iterator.batch(batch);

        for (const auto scaling : enum_values<scaling_type>())
        {
            iterator.scaling(scaling);
            UTEST_CHECK_EQUAL(iterator.scaling(), scaling);

            if (batch == 2)
            {
                UTEST_CHECK(!iterator.cache_targets(0U));
            }
            else
            {
                UTEST_CHECK(iterator.cache_targets(1U << 24));
            }

            const auto& stats                   = iterator.targets_stats();
            auto        expected_scaled_targets = expected_targets;
            UTEST_REQUIRE_NOTHROW(stats.scale(scaling, expected_scaled_targets));

            auto called = make_full_tensor<tensor_size_t>(make_dims(samples.size()), 0);
            UTEST_CHECK_NOTHROW(iterator.loop(
                [&](tensor_range_t range, size_t tnum, tensor4d_cmap_t targets)
                {
                    called.slice(range).full(1);
                    UTEST_CHECK_GREATER_EQUAL(tnum, 0U);
                    UTEST_CHECK_LESS(tnum, pool_t::max_size());
                    UTEST_CHECK_LESS_EQUAL(range.size(), batch);
                    UTEST_CHECK_GREATER_EQUAL(range.begin(), 0);
                    UTEST_CHECK_LESS_EQUAL(range.end(), samples.size());
                    UTEST_CHECK_CLOSE(targets, expected_scaled_targets.indexed(samples.slice(range)), epsilon);
                }));
            UTEST_CHECK_EQUAL(called, make_full_tensor<tensor_size_t>(make_dims(samples.size()), 1));
        }
    }
}

[[maybe_unused]] static void check_targets_stats(const dataset_t& dataset, const indices_t& expected_samples,
                                                 const tensor1d_t& expected_min, const tensor1d_t& expected_max,
                                                 const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev,
                                                 const scalar_t epsilon = 1e-12)
{
    const auto  samples  = arange(0, dataset.samples());
    const auto  iterator = targets_iterator_t{dataset, samples};
    const auto& stats    = iterator.targets_stats();

    UTEST_CHECK_EQUAL(stats.m_samples, expected_samples);
    UTEST_CHECK_CLOSE(stats.m_min, expected_min, epsilon);
    UTEST_CHECK_CLOSE(stats.m_max, expected_max, epsilon);
    UTEST_CHECK_CLOSE(stats.m_mean, expected_mean, epsilon);
    UTEST_CHECK_CLOSE(stats.m_stdev, expected_stdev, epsilon);
}
