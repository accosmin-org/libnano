#include "fixture/dataset.h"
#include <nano/dataset.h>

using namespace nano;

static auto make_features()
{
    return features_t
    {
        feature_t{"i8"}.scalar(feature_type::int8),
        feature_t{"i16"}.scalar(feature_type::int16),
        feature_t{"i32"}.scalar(feature_type::int32),
        feature_t{"i64"}.scalar(feature_type::int64),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"f64"}.scalar(feature_type::float64),

        feature_t{"ui8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)),
        feature_t{"ui16_struct"}.scalar(feature_type::uint16, make_dims(1, 1, 1)),
        feature_t{"ui32_struct"}.scalar(feature_type::uint32, make_dims(1, 2, 1)),
        feature_t{"ui64_struct"}.scalar(feature_type::uint64, make_dims(1, 1, 2)),

        feature_t{"sclass2"}.sclass(2),
        feature_t{"sclass10"}.sclass(10),

        feature_t{"mclass3"}.mclass(3),
    };
}

class fixture_dataset_t final : public dataset_t
{
public:

    fixture_dataset_t(tensor_size_t samples, features_t features, size_t target) :
        m_samples(samples),
        m_features(std::move(features)),
        m_target(target)
    {
    }

    void actually_do_load(bool do_load)
    {
        m_do_load = do_load;
    }

    static auto mask() { return make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }

    auto mask0() const { return m_target == 0U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    auto mask1() const { return m_target == 1U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xAA, 0xAA, 0xAA, 0x80); }
    auto mask2() const { return m_target == 2U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x92, 0x49, 0x24, 0x80); }
    auto mask3() const { return m_target == 3U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x88, 0x88, 0x88, 0x80); }
    auto mask4() const { return m_target == 4U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x84, 0x21, 0x08, 0x00); }
    auto mask5() const { return m_target == 5U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x82, 0x08, 0x20, 0x80); }
    auto mask6() const { return m_target == 6U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    auto mask7() const { return m_target == 7U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    auto mask8() const { return m_target == 8U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    auto mask9() const { return m_target == 9U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }
    auto mask10() const { return m_target == 10U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xAA, 0xAA, 0xAA, 0x80); }
    auto mask11() const { return m_target == 11U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x92, 0x49, 0x24, 0x80); }
    auto mask12() const { return m_target == 12U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x88, 0x88, 0x88, 0x80); }

    auto data0() const
    {
        return make_tensor<int8_t>(make_dims(25, 1, 1, 1),
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24);
    }
    auto data1() const
    {
        return m_target == 1U ?
            make_tensor<int16_t>(make_dims(25, 1, 1, 1),
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25) :
            make_tensor<int16_t>(make_dims(25, 1, 1, 1),
                1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0, 15, 0, 17, 0, 19, 0, 21, 0, 23, 0, 25);
    }
    auto data2() const
    {
        return m_target == 2U ?
            make_tensor<int32_t>(make_dims(25, 1, 1, 1),
                2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26) :
            make_tensor<int32_t>(make_dims(25, 1, 1, 1),
                2, 0, 0, 5, 0, 0, 8, 0, 0, 11, 0, 0, 14, 0, 0, 17, 0, 0, 20, 0, 0, 23, 0, 0, 26);
    }
    auto data3() const
    {
        return m_target == 3U ?
            make_tensor<int64_t>(make_dims(25, 1, 1, 1),
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27) :
            make_tensor<int64_t>(make_dims(25, 1, 1, 1),
                3, 0, 0, 0, 7, 0, 0, 0, 11, 0, 0, 0, 15, 0, 0, 0, 19, 0, 0, 0, 23, 0, 0, 0, 27);
    }
    auto data4() const
    {
        return m_target == 4U ?
            make_tensor<float>(make_dims(25, 1, 1, 1),
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28) :
            make_tensor<float>(make_dims(25, 1, 1, 1),
                4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 14, 0, 0, 0, 0, 19, 0, 0, 0, 0, 24, 0, 0, 0, 0);
    }
    auto data5() const
    {
        return m_target == 5U ?
            make_tensor<double>(make_dims(25, 1, 1, 1),
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29) :
            make_tensor<double>(make_dims(25, 1, 1, 1),
                5, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 29);
    }
    auto data6() const
    {
        return
            make_tensor<uint8_t>(make_dims(25, 2, 1, 2),
                0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
                0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0,
                0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0,
                0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0);
    }
    auto data7() const
    {
        return
            make_tensor<uint16_t>(make_dims(25, 1, 1, 1),
                0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3);
    }
    auto data8() const
    {
        return
            make_tensor<uint32_t>(make_dims(25, 1, 2, 1),
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3, 3, 4,
                4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0);
    }
    auto data9() const
    {
        return
            make_tensor<uint64_t>(make_dims(25, 1, 1, 2),
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 0, 1, 1, 2, 2, 3,
                3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6);
    }
    auto data10() const
    {
        return
            make_tensor<uint8_t>(make_dims(25),
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }
    auto data11() const
    {
        return m_target == 11U ?
            make_tensor<uint8_t>(make_dims(25),
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4) :
            make_tensor<uint8_t>(make_dims(25),
                0, 0, 0, 3, 0, 0, 6, 0, 0, 9, 0, 0, 2, 0, 0, 5, 0, 0, 8, 0, 0, 1, 0, 0, 4);
    }
    auto data12() const
    {
        return m_target == 12U ?
            make_tensor<uint8_t>(make_dims(25, 3),
                0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2,
                2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1,
                1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0) :
            make_tensor<uint8_t>(make_dims(25, 3),
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
                2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

private:

    void do_load() override
    {
        resize(m_samples, m_features, m_target);

        if (!m_do_load)
        {
            return;
        }

        const auto itarget = static_cast<tensor_size_t>(m_target);

        // scalars
        for (tensor_size_t feature = 0; feature < 6; ++ feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; sample += (itarget == feature) ? 1 : feature + 1)
            {
                this->set(sample, feature, sample + feature);
            }
        }

        // structured
        for (tensor_size_t feature = 6; feature < 10; ++ feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; ++ sample)
            {
                this->set(sample, feature, make_full_tensor<tensor_size_t>(
                    m_features[static_cast<size_t>(feature)].dims(), sample % feature));
            }
        }

        // single label
        for (tensor_size_t sample = 0, feature = 10; sample < m_samples; sample += (itarget == feature) ? 1 : 2)
        {
            this->set(sample, feature, sample % 2);
        }
        for (tensor_size_t sample = 0, feature = 11; sample < m_samples; sample += (itarget == feature) ? 1 : 3)
        {
            this->set(sample, feature, sample % 10);
        }

        // multi label
        for (tensor_size_t sample = 0, feature = 12; sample < m_samples; sample += (itarget == feature) ? 1 : 4)
        {
            this->set(sample, feature, make_full_tensor<uint8_t>(make_dims(3), sample % 3));
        }
    }

    tensor_size_t   m_samples{0};
    features_t      m_features;
    size_t          m_target;
    bool            m_do_load{true};
};

static auto make_dataset(tensor_size_t samples, const features_t& features, size_t target)
{
    auto dataset = fixture_dataset_t{samples, features, target};
    UTEST_CHECK_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), samples);
    return dataset;
}

UTEST_BEGIN_MODULE(test_dataset_memory)

UTEST_CASE(check_samples)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 100);
    auto dataset = make_dataset(samples.size(), features, string_t::npos);
    {
        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 0);

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 100);
        UTEST_CHECK_EQUAL(train_samples, ::nano::arange(0, 100));
    }
    {
        dataset.testing(make_range(0, 10));
        dataset.testing(make_range(20, 50));

        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 40);
        UTEST_CHECK_EQUAL(test_samples.slice(0, 10), ::nano::arange(0, 10));
        UTEST_CHECK_EQUAL(test_samples.slice(10, 40), ::nano::arange(20, 50));

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 60);
        UTEST_CHECK(train_samples.slice(0, 10) == ::nano::arange(10, 20));
        UTEST_CHECK(train_samples.slice(10, 60) == ::nano::arange(50, 100));
    }
    {
        dataset.no_testing();

        const auto test_samples = dataset.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 0);

        const auto train_samples = dataset.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 100);
        UTEST_CHECK_EQUAL(train_samples, ::nano::arange(0, 100));
    }
}

UTEST_CASE(dataset_target_NA)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), features, string_t::npos);

    UTEST_CHECK_EQUAL(dataset.features(), 13);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);

    check_inputs(dataset, 0, features[0U], dataset.data0(), dataset.mask0());
    check_inputs(dataset, 1, features[1U], dataset.data1(), dataset.mask1());
    check_inputs(dataset, 2, features[2U], dataset.data2(), dataset.mask2());
    check_inputs(dataset, 3, features[3U], dataset.data3(), dataset.mask3());
    check_inputs(dataset, 4, features[4U], dataset.data4(), dataset.mask4());
    check_inputs(dataset, 5, features[5U], dataset.data5(), dataset.mask5());
    check_inputs(dataset, 6, features[6U], dataset.data6(), dataset.mask6());
    check_inputs(dataset, 7, features[7U], dataset.data7(), dataset.mask7());
    check_inputs(dataset, 8, features[8U], dataset.data8(), dataset.mask8());
    check_inputs(dataset, 9, features[9U], dataset.data9(), dataset.mask9());
    check_inputs(dataset, 10, features[10U], dataset.data10(), dataset.mask10());
    check_inputs(dataset, 11, features[11U], dataset.data11(), dataset.mask11());
    check_inputs(dataset, 12, features[12U], dataset.data12(), dataset.mask12());
}

UTEST_CASE(dataset_target_0U)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), features, 0U);

    UTEST_CHECK_EQUAL(dataset.features(), 12);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::regression);

    check_target(dataset, features[0U], dataset.data0(), dataset.mask0());
    check_inputs(dataset, 0, features[1U], dataset.data1(), dataset.mask1());
    check_inputs(dataset, 1, features[2U], dataset.data2(), dataset.mask2());
    check_inputs(dataset, 2, features[3U], dataset.data3(), dataset.mask3());
    check_inputs(dataset, 3, features[4U], dataset.data4(), dataset.mask4());
    check_inputs(dataset, 4, features[5U], dataset.data5(), dataset.mask5());
    check_inputs(dataset, 5, features[6U], dataset.data6(), dataset.mask6());
    check_inputs(dataset, 6, features[7U], dataset.data7(), dataset.mask7());
    check_inputs(dataset, 7, features[8U], dataset.data8(), dataset.mask8());
    check_inputs(dataset, 8, features[9U], dataset.data9(), dataset.mask9());
    check_inputs(dataset, 9, features[10U], dataset.data10(), dataset.mask10());
    check_inputs(dataset, 10, features[11U], dataset.data11(), dataset.mask11());
    check_inputs(dataset, 11, features[12U], dataset.data12(), dataset.mask12());
}

UTEST_CASE(dataset_target_11U)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), features, 11U);

    UTEST_CHECK_EQUAL(dataset.features(), 12);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::sclassification);

    check_inputs(dataset, 0, features[0U], dataset.data0(), dataset.mask0());
    check_inputs(dataset, 1, features[1U], dataset.data1(), dataset.mask1());
    check_inputs(dataset, 2, features[2U], dataset.data2(), dataset.mask2());
    check_inputs(dataset, 3, features[3U], dataset.data3(), dataset.mask3());
    check_inputs(dataset, 4, features[4U], dataset.data4(), dataset.mask4());
    check_inputs(dataset, 5, features[5U], dataset.data5(), dataset.mask5());
    check_inputs(dataset, 6, features[6U], dataset.data6(), dataset.mask6());
    check_inputs(dataset, 7, features[7U], dataset.data7(), dataset.mask7());
    check_inputs(dataset, 8, features[8U], dataset.data8(), dataset.mask8());
    check_inputs(dataset, 9, features[9U], dataset.data9(), dataset.mask9());
    check_inputs(dataset, 10, features[10U], dataset.data10(), dataset.mask10());
    check_target(dataset, features[11U], dataset.data11(), dataset.mask11());
    check_inputs(dataset, 11, features[12U], dataset.data12(), dataset.mask12());
}

UTEST_CASE(dataset_target_12U)
{
    const auto features = make_features();
    const auto samples = ::nano::arange(0, 25);
    const auto dataset = make_dataset(samples.size(), features, 12U);

    UTEST_CHECK_EQUAL(dataset.features(), 12);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::mclassification);

    check_inputs(dataset, 0, features[0U], dataset.data0(), dataset.mask0());
    check_inputs(dataset, 1, features[1U], dataset.data1(), dataset.mask1());
    check_inputs(dataset, 2, features[2U], dataset.data2(), dataset.mask2());
    check_inputs(dataset, 3, features[3U], dataset.data3(), dataset.mask3());
    check_inputs(dataset, 4, features[4U], dataset.data4(), dataset.mask4());
    check_inputs(dataset, 5, features[5U], dataset.data5(), dataset.mask5());
    check_inputs(dataset, 6, features[6U], dataset.data6(), dataset.mask6());
    check_inputs(dataset, 7, features[7U], dataset.data7(), dataset.mask7());
    check_inputs(dataset, 8, features[8U], dataset.data8(), dataset.mask8());
    check_inputs(dataset, 9, features[9U], dataset.data9(), dataset.mask9());
    check_inputs(dataset, 10, features[10U], dataset.data10(), dataset.mask10());
    check_inputs(dataset, 11, features[11U], dataset.data11(), dataset.mask11());
    check_target(dataset, features[12U], dataset.data12(), dataset.mask12());
}

UTEST_CASE(invalid_feature_type)
{
    auto features = make_features();
    features[0].scalar(static_cast<feature_type>(-1));

    auto dataset = fixture_dataset_t{100, features, string_t::npos};
    UTEST_CHECK_THROW(dataset.load(), std::runtime_error);

    dataset.actually_do_load(false);
    UTEST_CHECK_NOTHROW(dataset.load());
    UTEST_CHECK_THROW(dataset.visit_inputs(0, [] (const auto&, const auto&, const auto&) {}), std::runtime_error);
}

UTEST_CASE(invalid_targets_type)
{
    auto features = make_features();
    features[0].scalar(static_cast<feature_type>(-1));

    auto dataset = fixture_dataset_t{100, features, string_t::npos};
    UTEST_CHECK_THROW(dataset.load(), std::runtime_error);

    dataset.actually_do_load(false);
    UTEST_CHECK_NOTHROW(dataset.load());
    UTEST_CHECK_THROW(dataset.visit_inputs(0, [] (const auto&, const auto&, const auto&) {}), std::runtime_error);
}

UTEST_END_MODULE()
