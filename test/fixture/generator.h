#include <nano/core/parallel.h>
#include <nano/dataset.h>
#include <nano/dataset/iterator.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::parallel;

static constexpr auto N   = std::numeric_limits<scalar_t>::quiet_NaN();
static constexpr auto Na  = std::numeric_limits<scalar_t>::quiet_NaN();
static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

static auto make_samples(const dataset_t& dataset)
{
    const auto samples = dataset.samples();

    return std::vector<indices_t>{arange(0, samples), arange(0, samples / 2), arange(samples / 2, samples)};
}

template <typename tgenerator>
static void add_generator(dataset_t& dataset)
{
    UTEST_CHECK_NOTHROW(dataset.add<tgenerator>());
}

template <typename tgenerator>
static void add_generator(dataset_t& dataset, indices_t features)
{
    UTEST_CHECK_NOTHROW(dataset.add<tgenerator>(std::move(features)));
}

template <typename tgenerator>
static void add_generator(dataset_t& dataset, indices_t features1, indices_t features2)
{
    UTEST_CHECK_NOTHROW(dataset.add<tgenerator>(std::move(features1), std::move(features2)));
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
    const auto shuffle = dataset.shuffled(samples, expected_feature);
    UTEST_REQUIRE_EQUAL(shuffle.size(), samples.size());
    UTEST_CHECK(std::is_permutation(shuffle.begin(), shuffle.end(), samples.begin()));
    // UTEST_CHECK_NOT_EQUAL(shuffle, samples);
    checker(expected.indexed(shuffle));

    const auto shuffle2 = dataset.shuffled(samples, expected_feature);
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
[[maybe_unused]] static void check_select(const dataset_t& dataset, tensor_size_t feature,
                                          const tensor_t<tstorage, tscalar, trank>& expected)
{
    auto iterator = select_iterator_t{dataset};

    const auto features = make_indices(feature);
    for (const auto& samples : make_samples(dataset))
    {
        check_select(iterator, samples, features, expected);
    }
}

[[maybe_unused]] static void check_flatten(const dataset_t& dataset, const tensor2d_t& expected_flatten,
                                           const indices_t& expected_column2features, scalar_t eps = 1e-12)
{
    UTEST_REQUIRE_EQUAL(dataset.columns(), expected_flatten.size<1>());
    UTEST_REQUIRE_EQUAL(dataset.columns(), expected_column2features.size());

    for (tensor_size_t column = 0; column < dataset.columns(); ++column)
    {
        UTEST_CHECK_EQUAL(dataset.column2feature(column), expected_column2features(column));
    }

    for (const auto& samples : make_samples(dataset))
    {
        auto iterator = flatten_iterator_t{dataset, samples};

        for (const auto batch : {1, 3, 8})
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

                auto called = make_full_tensor<tensor_size_t>(make_dims(samples.size()), 0);
                UTEST_CHECK_NOTHROW(iterator.loop(
                    [&](tensor_range_t range, size_t tnum, tensor2d_cmap_t flatten)
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
        }
    }
}

[[maybe_unused]] static void check_select_stats(const dataset_t& dataset, const indices_t& expected_sclass_features,
                                                const indices_t& expected_mclass_features,
                                                const indices_t& expected_scalar_features,
                                                const indices_t& expected_struct_features)
{
    UTEST_CHECK_EQUAL(dataset.sclass_features(), expected_sclass_features);
    UTEST_CHECK_EQUAL(dataset.mclass_features(), expected_mclass_features);
    UTEST_CHECK_EQUAL(dataset.scalar_features(), expected_scalar_features);
    UTEST_CHECK_EQUAL(dataset.struct_features(), expected_struct_features);

    const auto samples = arange(0, dataset.samples());

    auto features = std::vector<tensor_size_t>{};
    auto iterator = select_iterator_t{dataset, 1U};

    const auto op_sclass = [&](tensor_size_t feature, size_t, sclass_cmap_t) { features.push_back(feature); };
    const auto op_mclass = [&](tensor_size_t feature, size_t, mclass_cmap_t) { features.push_back(feature); };
    const auto op_scalar = [&](tensor_size_t feature, size_t, scalar_cmap_t) { features.push_back(feature); };
    const auto op_struct = [&](tensor_size_t feature, size_t, struct_cmap_t) { features.push_back(feature); };

    const auto make_features = [&]()
    { return map_tensor(features.data(), static_cast<tensor_size_t>(features.size())); };

    features.clear();
    UTEST_CHECK_NOTHROW(iterator.loop(samples, op_sclass));
    UTEST_CHECK_EQUAL(expected_sclass_features, make_features());

    features.clear();
    UTEST_CHECK_NOTHROW(iterator.loop(samples, op_mclass));
    UTEST_CHECK_EQUAL(expected_mclass_features, make_features());

    features.clear();
    UTEST_CHECK_NOTHROW(iterator.loop(samples, op_scalar));
    UTEST_CHECK_EQUAL(expected_scalar_features, make_features());

    features.clear();
    UTEST_CHECK_NOTHROW(iterator.loop(samples, op_struct));
    UTEST_CHECK_EQUAL(expected_struct_features, make_features());
}

[[maybe_unused]] static void check_flatten_stats0(const dataset_t& dataset, const indices_t& expected_samples,
                                                  const tensor1d_t& expected_min, const tensor1d_t& expected_max,
                                                  const tensor1d_t& expected_mean, const tensor1d_t& expected_stdev,
                                                  scalar_t eps = 1e-12)
{
    const auto samples = arange(0, dataset.samples());

    auto iterator = flatten_iterator_t{dataset, samples};
    for (const auto scaling : enum_values<scaling_type>())
    {
        iterator.batch(3);
        iterator.scaling(scaling);
        UTEST_CHECK_EQUAL(iterator.scaling(), scaling);

        const auto& stats = iterator.flatten_stats();
        UTEST_CHECK_EQUAL(stats.samples(), expected_samples);
        UTEST_CHECK_CLOSE(stats.min(), expected_min, eps);
        UTEST_CHECK_CLOSE(stats.max(), expected_max, eps);
        UTEST_CHECK_CLOSE(stats.mean(), expected_mean, eps);
        UTEST_CHECK_CLOSE(stats.stdev(), expected_stdev, eps);
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
                                           scalar_t eps = 1e-12)
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
            if (std::holds_alternative<scalar_stats_t>(stats))
            {
                UTEST_REQUIRE_NOTHROW(std::get<scalar_stats_t>(stats).scale(scaling, expected_scaled_targets));
            }

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
                    UTEST_CHECK_CLOSE(targets, expected_scaled_targets.indexed(samples.slice(range)), eps);
                }));
            UTEST_CHECK_EQUAL(called, make_full_tensor<tensor_size_t>(make_dims(samples.size()), 1));
        }
    }
}

[[maybe_unused]] static void check_targets_sclass_stats(const dataset_t&  dataset,
                                                        const indices_t&  expected_class_counts,
                                                        const tensor1d_t& expected_sample_weights, scalar_t eps = 1e-12)
{
    const auto samples = arange(0, dataset.samples());

    auto iterator = targets_iterator_t{dataset, samples};
    for (const auto batch : {1, 3, 8})
    {
        iterator.batch(batch);

        for (const auto scaling : enum_values<scaling_type>())
        {
            iterator.scaling(scaling);
            UTEST_CHECK_EQUAL(iterator.scaling(), scaling);

            auto stats = iterator.targets_stats();
            UTEST_REQUIRE_NOTHROW(std::get<sclass_stats_t>(stats));
            UTEST_CHECK_EQUAL(std::get<sclass_stats_t>(stats).class_counts(), expected_class_counts);
            UTEST_CHECK_CLOSE(dataset.sample_weights(samples, stats), expected_sample_weights, eps);

            stats = sclass_stats_t{42};
            UTEST_CHECK_THROW(dataset.sample_weights(samples, stats), std::runtime_error);

            stats = mclass_stats_t{expected_class_counts.size()};
            UTEST_CHECK_THROW(dataset.sample_weights(samples, stats), std::runtime_error);
        }
    }
}

[[maybe_unused]] static void check_targets_mclass_stats(const dataset_t&  dataset,
                                                        const indices_t&  expected_class_counts,
                                                        const tensor1d_t& expected_sample_weights, scalar_t eps = 1e-12)
{
    const auto samples = arange(0, dataset.samples());

    auto iterator = targets_iterator_t{dataset, samples};
    for (const auto batch : {1, 3, 8})
    {
        iterator.batch(batch);

        for (const auto scaling : enum_values<scaling_type>())
        {
            iterator.scaling(scaling);
            UTEST_CHECK_EQUAL(iterator.scaling(), scaling);

            auto stats = iterator.targets_stats();
            UTEST_REQUIRE_NOTHROW(std::get<mclass_stats_t>(stats));
            UTEST_CHECK_EQUAL(std::get<mclass_stats_t>(stats).class_counts(), expected_class_counts);
            UTEST_CHECK_CLOSE(dataset.sample_weights(samples, stats), expected_sample_weights, eps);

            stats = mclass_stats_t{42};
            UTEST_CHECK_THROW(dataset.sample_weights(samples, stats), std::runtime_error);

            stats = sclass_stats_t{expected_class_counts.size() / 2};
            UTEST_CHECK_THROW(dataset.sample_weights(samples, stats), std::runtime_error);
        }
    }
}

[[maybe_unused]] static void check_targets_scalar_stats(const dataset_t& dataset, const indices_t& expected_samples,
                                                        const tensor1d_t& expected_min, const tensor1d_t& expected_max,
                                                        const tensor1d_t& expected_mean,
                                                        const tensor1d_t& expected_stdev, scalar_t eps = 1e-12)
{
    tensor1d_t expected_sample_weights = tensor1d_t{dataset.samples()};
    expected_sample_weights.full(1.0);

    const auto samples = arange(0, dataset.samples());

    auto iterator = targets_iterator_t{dataset, samples};
    for (const auto batch : {1, 3, 8})
    {
        iterator.batch(batch);

        for (const auto scaling : enum_values<scaling_type>())
        {
            iterator.scaling(scaling);
            UTEST_CHECK_EQUAL(iterator.scaling(), scaling);

            const auto& stats = iterator.targets_stats();
            UTEST_REQUIRE_NOTHROW(std::get<scalar_stats_t>(stats));
            UTEST_CHECK_EQUAL(std::get<scalar_stats_t>(stats).samples(), expected_samples);
            UTEST_CHECK_CLOSE(std::get<scalar_stats_t>(stats).min(), expected_min, eps);
            UTEST_CHECK_CLOSE(std::get<scalar_stats_t>(stats).max(), expected_max, eps);
            UTEST_CHECK_CLOSE(std::get<scalar_stats_t>(stats).mean(), expected_mean, eps);
            UTEST_CHECK_CLOSE(std::get<scalar_stats_t>(stats).stdev(), expected_stdev, eps);
            UTEST_CHECK_CLOSE(dataset.sample_weights(samples, stats), expected_sample_weights, eps);
        }
    }
}
