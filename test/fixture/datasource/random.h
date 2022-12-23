#include <nano/datasource.h>

using namespace nano;

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
