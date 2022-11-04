#include "fixture/datasource.h"

using namespace nano;

static auto make_features()
{
    return features_t{
        feature_t{"mclass0"}.mclass(strings_t{"m00", "m01", "m02"}),
        feature_t{"mclass1"}.mclass(strings_t{"m10", "m11", "m12", "m13"}),
        feature_t{"sclass0"}.sclass(strings_t{"s00", "s01", "s02"}),
        feature_t{"sclass1"}.sclass(strings_t{"s10", "s11"}),
        feature_t{"sclass2"}.sclass(strings_t{"s20", "s21"}),
        feature_t{"scalar0"}.scalar(feature_type::int16),
        feature_t{"scalar1"}.scalar(feature_type::int64),
        feature_t{"scalar2"}.scalar(feature_type::int8),
        feature_t{"struct0"}.scalar(feature_type::uint8, make_dims(1, 2, 2)),
        feature_t{"struct1"}.scalar(feature_type::uint16, make_dims(2, 1, 3)),
        feature_t{"struct2"}.scalar(feature_type::uint32, make_dims(3, 1, 1)),
    };
}

class fixture_datasource_t final : public datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples, const size_t target = string_t::npos)
        : datasource_t("fixture")
        , m_samples(samples)
        , m_features(make_features())
        , m_target(target)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

private:
    void set_mclass(tensor_size_t ifeature, tensor_size_t classes, tensor_size_t modulo)
    {
        tensor_mem_t<tensor_size_t, 1> hits(classes);
        hits.full(0);
        for (tensor_size_t sample = 0; sample < m_samples; sample += modulo)
        {
            hits(0) = sample % 2;
            hits(1) = 1 - (sample % 2);
            hits(2) = ((sample % 6) == 0) ? 1 : 0;
            set(sample, ifeature, hits);
        }
    }

    void set_sclass(tensor_size_t ifeature, tensor_size_t classes, tensor_size_t modulo)
    {
        for (tensor_size_t sample = 0; sample < m_samples; sample += modulo)
        {
            set(sample, ifeature, (sample + modulo) % classes);
        }
    }

    void set_scalar(tensor_size_t ifeature, tensor_size_t modulo)
    {
        for (tensor_size_t sample = 0; sample < m_samples; sample += modulo)
        {
            set(sample, ifeature, sample - modulo);
        }
    }

    void set_struct(tensor_size_t ifeature, const tensor3d_dims_t& dims, tensor_size_t modulo)
    {
        tensor_mem_t<tensor_size_t, 3> values(dims);
        for (tensor_size_t sample = 0; sample < m_samples; sample += modulo)
        {
            values.full(sample);
            values(0) = sample + 1;
            set(sample, ifeature, values);
        }
    }

    void do_load() override
    {
        resize(m_samples, m_features, m_target);

        set_mclass(0, 3, 1);
        set_mclass(1, 4, 2);

        set_sclass(2, 3, 2);
        set_sclass(3, 2, 1);
        set_sclass(4, 2, 2);

        set_scalar(5, 1);
        set_scalar(6, 2);
        set_scalar(7, 3);

        set_struct(8, make_dims(1, 2, 2), 1);
        set_struct(9, make_dims(2, 1, 3), 2);
        set_struct(10, make_dims(3, 1, 1), 3);
    }

    tensor_size_t m_samples{0};
    features_t    m_features;
    size_t        m_target;
};

static auto make_datasource(const tensor_size_t samples, const size_t target)
{
    auto datasource = fixture_datasource_t{samples, target};
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}
