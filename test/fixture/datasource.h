#include <nano/datasource.h>
#include <utest/utest.h>

using namespace nano;

template <typename tscalar, size_t trank>
static auto check_inputs(const datasource_t& datasource, tensor_size_t index, const feature_t& gt_feature,
                         const tensor_mem_t<tscalar, trank>& gt_data, const mask_cmap_t& gt_mask)
{
    const auto visitor = [&](const auto& feature, const auto& data, const auto& mask)
    {
        UTEST_CHECK_EQUAL(feature, gt_feature);
        if constexpr (std::is_same<decltype(data), const tensor_cmap_t<tscalar, trank>&>::value)
        {
            UTEST_CHECK_CLOSE(data, gt_data, 1e-12);
            UTEST_CHECK_EQUAL(mask, gt_mask);
        }
        else
        {
            UTEST_CHECK(false);
        }
    };

    datasource.visit_inputs(index, visitor);
}

template <typename tscalar, size_t trank>
static auto check_target(const datasource_t& datasource, const feature_t& gt_feature,
                         const tensor_mem_t<tscalar, trank>& gt_data, const mask_cmap_t& gt_mask)
{
    const auto visitor = [&](const auto& feature, const auto& data, const auto& mask)
    {
        UTEST_CHECK_EQUAL(feature, gt_feature);
        if constexpr (std::is_same<decltype(data), const tensor_cmap_t<tscalar, trank>&>::value)
        {
            UTEST_CHECK_CLOSE(data, gt_data, 1e-12);
            UTEST_CHECK_EQUAL(mask, gt_mask);
        }
        else
        {
            UTEST_CHECK(false);
        }
    };

    datasource.visit_target(visitor);
}

[[maybe_unused]] static auto make_random_hits(const tensor_size_t samples, const tensor_size_t features,
                                              const size_t target)
{
    auto hits = make_random_tensor<int8_t>(make_dims(samples, features), 0, 1);

    if (target != string_t::npos)
    {
        hits.matrix().col(static_cast<tensor_size_t>(target)).array() = 1;
    }

    return hits;
}

[[maybe_unused]] static auto make_all_hits(const tensor_size_t samples, const tensor_size_t features)
{
    return make_full_tensor<int8_t>(make_dims(samples, features), 1);
}

class random_datasource_t : public datasource_t
{
public:
    random_datasource_t(const tensor_size_t samples, features_t features, const size_t target,
                        tensor_mem_t<int8_t, 2> hits)
        : datasource_t("random")
        , m_samples(samples)
        , m_features(std::move(features))
        , m_target(target)
        , m_hits(std::move(hits))
    {
        assert(m_hits.rows() == m_samples);
        assert(m_hits.cols() == static_cast<tensor_size_t>(m_features.size()));
    }

    rdatasource_t clone() const override { return std::make_unique<random_datasource_t>(*this); }

    const auto& hits() const { return m_hits; }

protected:
    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
    void set_fvalues(const tensor_size_t feature, const tensor_t<tstorage, tscalar, trank>& fvalues)
    {
        if constexpr (trank == 1)
        {
            for (tensor_size_t sample = 0; sample < m_samples; ++sample)
            {
                if (m_hits(sample, feature) != 0)
                {
                    set(sample, feature, fvalues(sample));
                }
            }
        }
        else
        {
            for (tensor_size_t sample = 0; sample < m_samples; ++sample)
            {
                if (m_hits(sample, feature) != 0)
                {
                    set(sample, feature, fvalues.tensor(sample));
                }
            }
        }
    }

    void do_load() override
    {
        resize(m_samples, m_features, m_target);

        tensor_size_t ifeature = 0;
        for (const auto& feature : m_features)
        {
            switch (feature.type())
            {
            case feature_type::sclass:
                set_fvalues(ifeature,
                            make_random_tensor<int8_t>(make_dims(m_samples), tensor_size_t{0}, feature.classes() - 1));
                break;

            case feature_type::mclass:
                set_fvalues(ifeature, make_random_tensor<int8_t>(make_dims(m_samples, feature.classes()), 0, 1));
                break;

            case feature_type::uint8:
            case feature_type::uint16:
            case feature_type::uint32:
            case feature_type::uint64:
                set_fvalues(ifeature, make_random_tensor<uint8_t>(cat_dims(m_samples, feature.dims()), 0, 13));
                break;

            case feature_type::int8:
            case feature_type::int16:
            case feature_type::int32:
            case feature_type::int64:
                set_fvalues(ifeature, make_random_tensor<int8_t>(cat_dims(m_samples, feature.dims()), -11, +17));
                break;

            case feature_type::float32:
                set_fvalues(ifeature, make_random_tensor<float>(cat_dims(m_samples, feature.dims()), -3.0, +2.9));
                break;

            default:
                set_fvalues(ifeature, make_random_tensor<scalar_t>(cat_dims(m_samples, feature.dims()), -1.2, +1.3));
                break;
            }
            ++ifeature;
        }
    }

private:
    tensor_size_t           m_samples{0};
    features_t              m_features;
    size_t                  m_target{0U};
    tensor_mem_t<int8_t, 2> m_hits;
};
