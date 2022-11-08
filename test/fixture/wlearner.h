#include "fixture/dataset.h"
#include "fixture/datasource.h"
#include "fixture/estimator.h"
#include "fixture/loss.h"
#include <nano/dataset/iterator.h>
#include <nano/wlearner.h>

using namespace nano;

static auto make_features()
{
    return features_t{
        feature_t{"sclass0"}.sclass(strings_t{"s10", "s11"}),
        feature_t{"sclass1"}.sclass(strings_t{"s00", "s01", "s02"}),
        feature_t{"sclass2"}.sclass(strings_t{"s20", "s21"}),
        feature_t{"mclass0"}.mclass(strings_t{"m00", "m01", "m02"}),
        feature_t{"mclass1"}.mclass(strings_t{"m10", "m11", "m12", "m13"}),
        feature_t{"scalar0"}.scalar(feature_type::float32),
        feature_t{"scalar1"}.scalar(feature_type::float64),
        feature_t{"scalar2"}.scalar(feature_type::int16),
        feature_t{"struct0"}.scalar(feature_type::uint64, make_dims(1, 2, 2)),
        feature_t{"struct1"}.scalar(feature_type::float32, make_dims(2, 1, 3)),
        feature_t{"struct2"}.scalar(feature_type::int64, make_dims(3, 1, 1)),
        feature_t{"target"}.scalar(feature_type::float64),
    };
}

class wlearner_datasource_t : public random_datasource_t
{
public:
    explicit wlearner_datasource_t(const tensor_size_t samples, const tensor_size_t groups)
        : random_datasource_t(samples, make_features(), 11U, make_random_hits(samples, 12, 11U))
        , m_cluster(samples, groups)
    {
    }

    void assign(const tensor_size_t sample, const tensor_size_t group) { m_cluster.assign(sample, group); }

    const auto& expected_cluster() const { return m_cluster; }

private:
    cluster_t m_cluster;
};

static auto make_features_too_few()
{
    auto features = make_features();
    features.erase(features.begin() + 1U);
    return features;
}

static auto make_features_too_many()
{
    auto       features = make_features();
    const auto feature  = features[2U];
    features.insert(features.begin() + 1U, feature);
    return features;
}

static auto make_features_invalid_target()
{
    auto features = make_features();
    features.rbegin()->scalar(feature_type::float64, make_dims(2, 1, 1));
    return features;
}

[[maybe_unused]] static auto make_features_all_continuous()
{
    auto features = make_features();
    features.erase(features.begin(), features.begin() + 5U);
    return features;
}

[[maybe_unused]] static auto make_features_all_discrete()
{
    auto features = make_features();
    features.erase(features.begin() + 5U, features.begin() + 11U);
    return features;
}

template <typename tdatasource, typename... targs>
static auto make_datasource(const tensor_size_t samples, const targs... args)
{
    auto datasource = tdatasource{samples, args...};
    UTEST_REQUIRE_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}

static auto make_random_datasource(const features_t& features, const tensor_size_t samples = 100)
{
    const auto target = features.size() - 1U;
    const auto hits   = make_random_hits(samples, static_cast<tensor_size_t>(features.size()), target);

    auto datasource = random_datasource_t{samples, features, target, hits};
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}

inline auto make_cut_samples(const dataset_t& dataset)
{
    return arange(1 * dataset.samples() / 10, 9 * dataset.samples() / 10);
}

inline auto make_all_samples(const dataset_t& dataset)
{
    return arange(0, dataset.samples());
}

inline auto make_targets(const dataset_t& dataset)
{
    const auto samples = make_all_samples(dataset);

    tensor4d_t targets(cat_dims(dataset.samples(), dataset.target_dims()));

    targets_iterator_t it(dataset, samples, 1U);
    it.loop([&](const tensor_range_t range, size_t, tensor4d_cmap_t _targets) { targets.slice(range) = _targets; });

    return targets;
}

inline auto make_residuals(const dataset_t& dataset, const loss_t& loss)
{
    const auto samples = make_all_samples(dataset);
    const auto outputs = make_full_tensor<scalar_t>(cat_dims(dataset.samples(), dataset.target_dims()), 0.0);
    const auto targets = make_targets(dataset);

    tensor4d_t residuals(outputs.dims());
    loss.vgrad(targets, outputs, residuals);
    return residuals;
}

inline auto check_fit(wlearner_t& wlearner, const dataset_t& dataset)
{
    const auto loss      = make_loss();
    const auto samples   = make_cut_samples(dataset);
    const auto residuals = make_residuals(dataset, *loss);

    auto fit_score = wlearner_t::no_fit_score();
    UTEST_REQUIRE_NOTHROW(fit_score = wlearner.fit(dataset, samples, residuals));
    UTEST_REQUIRE(std::isfinite(fit_score));
    return fit_score;
}

inline auto check_fit(wlearner_t& wlearner, const datasource_t& datasource)
{
    return check_fit(wlearner, make_dataset(datasource));
}

inline void check_no_fit(wlearner_t& wlearner, const dataset_t& dataset)
{
    const auto loss      = make_loss();
    const auto samples   = make_cut_samples(dataset);
    const auto residuals = make_residuals(dataset, *loss);

    auto fit_score = wlearner_t::no_fit_score();
    UTEST_CHECK_NOTHROW(fit_score = wlearner.fit(dataset, samples, residuals));
    UTEST_CHECK(std::isfinite(fit_score));
    UTEST_CHECK_EQUAL(fit_score, wlearner_t::no_fit_score());
}

inline void check_no_fit(wlearner_t& wlearner, const datasource_t& datasource)
{
    check_no_fit(wlearner, make_dataset(datasource));
}

inline void check_split(const wlearner_t& wlearner, const dataset_t& dataset, const cluster_t& expected_cluster)
{
    const auto samples = make_all_samples(dataset);

    cluster_t cluster;
    UTEST_CHECK_NOTHROW(cluster = wlearner.split(dataset, samples));

    UTEST_REQUIRE_EQUAL(cluster.samples(), dataset.samples());
    UTEST_REQUIRE_EQUAL(cluster.samples(), expected_cluster.samples());

    UTEST_REQUIRE_EQUAL(cluster.groups(), expected_cluster.groups());
    for (tensor_size_t group = 0; group < expected_cluster.groups(); ++group)
    {
        UTEST_REQUIRE_EQUAL(cluster.count(group), expected_cluster.count(group));
        UTEST_CHECK_EQUAL(cluster.indices(group), expected_cluster.indices(group));
    }
}

inline void check_split(const wlearner_t& wlearner, const datasource_t& datasource, const cluster_t& expected_cluster)
{
    check_split(wlearner, make_dataset(datasource), expected_cluster);
}

inline void check_split_throws(const wlearner_t& wlearner, const dataset_t& dataset)
{
    const auto samples = make_all_samples(dataset);

    cluster_t cluster;
    UTEST_CHECK_THROW(cluster = wlearner.split(dataset, samples), std::runtime_error);
}

inline void check_split_throws(const wlearner_t& wlearner, const datasource_t& datasource)
{
    check_split_throws(wlearner, make_dataset(datasource));
}

template <typename... tdatasources>
inline void check_split_throws(const wlearner_t& wlearner, const datasource_t& datasource,
                               const tdatasources&... datasources)
{
    check_split_throws(wlearner, datasource);
    check_split_throws(wlearner, datasources...);
}

inline void check_predict(const wlearner_t& wlearner, const dataset_t& dataset, const cluster_t& expected_cluster)
{
    const auto all_targets = make_targets(dataset);

    for (const auto& samples : {make_cut_samples(dataset), make_all_samples(dataset)})
    {
        tensor4d_t outputs;
        UTEST_REQUIRE_NOTHROW(outputs = wlearner.predict(dataset, samples));

        const auto targets = all_targets.indexed(samples);
        UTEST_REQUIRE_EQUAL(outputs.dims(), targets.dims());

        for (tensor_size_t i = 0; i < samples.size(); ++i)
        {
            if (expected_cluster.group(samples(i)) < 0)
            {
                UTEST_CHECK_CLOSE(outputs.tensor(i).min(), 0.0, 1e-15);
                UTEST_CHECK_CLOSE(outputs.tensor(i).max(), 0.0, 1e-15);
            }
            else
            {
                UTEST_CHECK_CLOSE(outputs.tensor(i), targets.tensor(i), 1e-8);
            }
        }
    }
}

inline void check_predict(const wlearner_t& wlearner, const datasource_t& datasource, const cluster_t& expected_cluster)
{
    check_predict(wlearner, make_dataset(datasource), expected_cluster);
}

inline void check_predict_throws(const wlearner_t& wlearner, const dataset_t& dataset)
{
    for (const auto& samples : {make_cut_samples(dataset), make_all_samples(dataset)})
    {
        UTEST_CHECK_THROW(wlearner.predict(dataset, samples), std::runtime_error);
    }
}

inline void check_predict_throws(const wlearner_t& wlearner, const datasource_t& datasource)
{
    check_predict_throws(wlearner, make_dataset(datasource));
}

template <typename... tdatasources>
inline void check_predict_throws(const wlearner_t& wlearner, const datasource_t& datasource,
                                 const tdatasources&... datasources)
{
    check_predict_throws(wlearner, datasource);
    check_predict_throws(wlearner, datasources...);
}

inline void check_scale(wlearner_t& wlearner, const dataset_t& dataset, const cluster_t& expected_cluster)
{
    for (const auto& samples : {make_cut_samples(dataset), make_all_samples(dataset)})
    {
        tensor4d_t outputs;
        UTEST_CHECK_NOTHROW(outputs = wlearner.predict(dataset, samples));
        {
            const auto scale   = make_full_vector<scalar_t>(1, 2.0);
            const auto unscale = make_full_vector<scalar_t>(1, 0.5);

            tensor4d_t outputs_scaled;
            UTEST_CHECK_NOTHROW(wlearner.scale(scale));
            UTEST_CHECK_NOTHROW(outputs_scaled = wlearner.predict(dataset, samples));
            UTEST_CHECK_CLOSE(outputs.array() * scale(0), outputs_scaled.array(), 1e-8);

            UTEST_CHECK_NOTHROW(wlearner.scale(unscale));
        }
        if (expected_cluster.groups() != 1)
        {
            const auto scale = make_random_vector<scalar_t>(expected_cluster.groups(), 2.0, 3.0);

            tensor4d_t outputs_scaled;
            UTEST_CHECK_NOTHROW(wlearner.scale(scale));
            UTEST_CHECK_NOTHROW(outputs_scaled = wlearner.predict(dataset, samples));
            for (tensor_size_t s = 0; s < samples.size(); ++s)
            {
                const auto group  = expected_cluster.group(samples(s));
                const auto factor = (group < 0) ? 1.0 : scale(group);
                UTEST_CHECK_CLOSE(outputs.array(s) * factor, outputs_scaled.array(s), 1e-8);
            }
        }
    }
}

inline void check_scale(wlearner_t& wlearner, const datasource_t& datasource, const cluster_t& expected_cluster)
{
    check_scale(wlearner, make_dataset(datasource), expected_cluster);
}

template <typename twlearner, typename tdatasource, typename... tinvalid_datasources>
void check_wlearner(const tdatasource& datasource0, const tinvalid_datasources&... datasourceXs)
{
    const auto datasourceX1 = make_random_datasource(make_features_too_few());
    const auto datasourceX2 = make_random_datasource(make_features_too_many());
    const auto datasourceX3 = make_random_datasource(make_features_invalid_target());

    const auto& expected_cluster = datasource0.expected_cluster();

    // no compatible features, so fitting will not work
    auto wlearner = twlearner{};
    check_no_fit(wlearner, datasourceXs...);

    // not fitting yet, so the weak learner should not be usable before fitting
    check_predict_throws(wlearner, datasource0, datasourceX1, datasourceX2, datasourceX3, datasourceXs...);
    check_split_throws(wlearner, datasource0, datasourceX1, datasourceX2, datasourceX3, datasourceXs...);

    // check fitting
    const auto score = check_fit(wlearner, datasource0);
    UTEST_CHECK_CLOSE(score, 0.0, 1e-8);
    datasource0.check_wlearner(wlearner);

    // check prediction
    check_predict(wlearner, datasource0, expected_cluster);
    check_predict_throws(wlearner, datasourceX1, datasourceX2, datasourceX3, datasourceXs...);

    // check splitting
    check_split(wlearner, datasource0, expected_cluster);
    check_split_throws(wlearner, datasourceX1, datasourceX2, datasourceX3, datasourceXs...);

    // check model loading and saving from and to binary streams
    const auto iwlearner = check_stream(wlearner);
    datasource0.check_wlearner(iwlearner);

    // check scaling
    check_scale(wlearner, datasource0, expected_cluster);
}
