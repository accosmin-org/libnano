#include <fstream>
#include <nano/core/stream.h>
#include <nano/feature.h>
#include <utest/utest.h>

using namespace nano;

static void check_stream(const feature_t& feature)
{
    {
        std::ofstream stream;
        UTEST_CHECK_THROW(feature.write(stream), std::runtime_error);
    }
    {
        feature_t     xfeature;
        std::ifstream stream;
        UTEST_CHECK_THROW(xfeature.read(stream), std::runtime_error);
    }
    {
        std::ostringstream ostream;
        UTEST_CHECK_NOTHROW(::nano::write(ostream, feature));

        feature_t xfeature;
        UTEST_CHECK_NOT_EQUAL(feature, xfeature);
        std::istringstream istream(ostream.str());
        UTEST_CHECK_NOTHROW(::nano::read(istream, xfeature));
        UTEST_CHECK_EQUAL(feature, xfeature);
    }
}

UTEST_BEGIN_MODULE(test_feature)

UTEST_CASE(_default)
{
    feature_t feature;
    UTEST_CHECK_EQUAL(feature.valid(), false);
    UTEST_CHECK_EQUAL(feature.task(), task_type::unsupervised);

    feature = feature_t{"feature"};
    UTEST_CHECK_EQUAL(feature.valid(), true);
    UTEST_CHECK_EQUAL(feature.dims(), make_dims(1, 1, 1));
    UTEST_CHECK_EQUAL(feature.type(), feature_type::float32);
    UTEST_CHECK_EQUAL(feature.task(), task_type::regression);
}

UTEST_CASE(task_type)
{
    {
        const auto feature = feature_t{};
        UTEST_CHECK(!feature.is_sclass());
        UTEST_CHECK(!feature.is_mclass());
        UTEST_CHECK(!feature.is_scalar());
        UTEST_CHECK(!feature.is_struct());
        UTEST_CHECK_EQUAL(feature.task(), task_type::unsupervised);
    }
    {
        const auto feature = feature_t{"feature"}.sclass(7);
        UTEST_CHECK(feature.is_sclass());
        UTEST_CHECK(!feature.is_mclass());
        UTEST_CHECK(!feature.is_scalar());
        UTEST_CHECK(!feature.is_struct());
        UTEST_CHECK_EQUAL(feature.task(), task_type::sclassification);
    }
    {
        const auto feature = feature_t{"feature"}.mclass(7);
        UTEST_CHECK(!feature.is_sclass());
        UTEST_CHECK(feature.is_mclass());
        UTEST_CHECK(!feature.is_scalar());
        UTEST_CHECK(!feature.is_struct());
        UTEST_CHECK_EQUAL(feature.task(), task_type::mclassification);
    }
    {
        const auto feature = feature_t{"feature"};
        UTEST_CHECK(!feature.is_sclass());
        UTEST_CHECK(!feature.is_mclass());
        UTEST_CHECK(feature.is_scalar());
        UTEST_CHECK(!feature.is_struct());
        UTEST_CHECK_EQUAL(feature.task(), task_type::regression);
    }
    {
        const auto feature = feature_t{"feature"}.scalar();
        UTEST_CHECK(!feature.is_sclass());
        UTEST_CHECK(!feature.is_mclass());
        UTEST_CHECK(feature.is_scalar());
        UTEST_CHECK(!feature.is_struct());
        UTEST_CHECK_EQUAL(feature.task(), task_type::regression);
    }
    {
        const auto feature = feature_t{"feature"}.scalar(feature_type::float32, make_dims(1, 1, 2));
        UTEST_CHECK(!feature.is_sclass());
        UTEST_CHECK(!feature.is_mclass());
        UTEST_CHECK(!feature.is_scalar());
        UTEST_CHECK(feature.is_struct());
        UTEST_CHECK_EQUAL(feature.dims(), make_dims(1, 1, 2));
        UTEST_CHECK_EQUAL(feature.task(), task_type::regression);
    }
    {
        const auto feature = feature_t{"feature"}.scalar(feature_type::float64, make_dims(3, 2, 1));
        UTEST_CHECK(!feature.is_sclass());
        UTEST_CHECK(!feature.is_mclass());
        UTEST_CHECK(!feature.is_scalar());
        UTEST_CHECK(feature.is_struct());
        UTEST_CHECK_EQUAL(feature.dims(), make_dims(3, 2, 1));
        UTEST_CHECK_EQUAL(feature.task(), task_type::regression);
    }
}

UTEST_CASE(compare)
{
    const auto make_feature_cont =
        [](const string_t& name, feature_type type = feature_type::float32, tensor3d_dims_t dims = make_dims(1, 1, 1))
    {
        auto feature = feature_t{name}.scalar(type, dims);
        UTEST_CHECK_EQUAL(feature.type(), type);
        return feature;
    };

    const auto make_feature_cate = [](const string_t& name, feature_type type = feature_type::sclass)
    {
        assert(type == feature_type::sclass || type == feature_type::mclass);
        auto feature = feature_t{name};
        switch (type)
        {
        case feature_type::sclass: feature.sclass(strings_t{"cate0", "cate1", "cate2"}); break;
        default: feature.mclass(strings_t{"cate0", "cate1", "cate2"}); break;
        }
        UTEST_CHECK_EQUAL(feature.type(), type);
        return feature;
    };

    const auto to_string = [](const feature_t& feature)
    {
        std::stringstream stream;
        stream << feature;
        return stream.str();
    };

    UTEST_CHECK_EQUAL(make_feature_cont("f"), make_feature_cont("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont("gf"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont("f", feature_type::float64));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont("f", feature_type::float32, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont("f")), "name=f,type=float32,dims=1x1x1,labels[]");

    UTEST_CHECK_EQUAL(make_feature_cate("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate("f"), make_feature_cate("x"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate("f")), "name=f,type=sclass,dims=1x1x1,labels[cate0,cate1,cate2]");

    UTEST_CHECK_NOT_EQUAL(feature_t{"f"}.sclass(strings_t{"label1", "label2"}),
                          feature_t{"f"}.sclass(strings_t{"label2", "label1"}));

    UTEST_CHECK_NOT_EQUAL(feature_t{"f"}.sclass(strings_t{"label1", "label2"}),
                          feature_t{"f"}.sclass(strings_t{"label1", "label2", "label3"}));

    UTEST_CHECK_EQUAL(feature_t{"f"}.sclass(strings_t{"label1", "label2"}),
                      feature_t{"f"}.sclass(strings_t{"label1", "label2"}));
}

UTEST_CASE(feature_info)
{
    {
        const auto info = feature_info_t{};
        UTEST_CHECK_CLOSE(info.importance(), 0.0, 1e-12);
    }
    {
        const auto info = feature_info_t{7, 13, 42.0};
        UTEST_CHECK_EQUAL(info.feature(), 7);
        UTEST_CHECK_EQUAL(info.count(), 13);
        UTEST_CHECK_CLOSE(info.importance(), 42.0, 1e-12);
    }
    {
        auto info = feature_info_t{};
        UTEST_CHECK_NOTHROW(info.importance(37.5));
        UTEST_CHECK_CLOSE(info.importance(), 37.5, 1e-12);
    }
    {
        auto infos = feature_infos_t{
            feature_info_t{5, 1, 45.0},
            feature_info_t{6, 2, 36.0},
            feature_info_t{4, 7, 41.0}
        };

        feature_info_t::sort_by_index(infos);
        UTEST_REQUIRE_EQUAL(infos.size(), 3U);
        UTEST_CHECK_EQUAL(infos[0].feature(), 4);
        UTEST_CHECK_EQUAL(infos[1].feature(), 5);
        UTEST_CHECK_EQUAL(infos[2].feature(), 6);

        feature_info_t::sort_by_importance(infos);
        UTEST_REQUIRE_EQUAL(infos.size(), 3U);
        UTEST_CHECK_EQUAL(infos[0].feature(), 5);
        UTEST_CHECK_EQUAL(infos[1].feature(), 4);
        UTEST_CHECK_EQUAL(infos[2].feature(), 6);
    }
}

UTEST_CASE(stream_feature)
{
    check_stream(feature_t{"f32"}.scalar(feature_type::float32, make_dims(1, 1, 1)));
    check_stream(feature_t{"t64"}.scalar(feature_type::float64, make_dims(3, 2, 4)));
    check_stream(feature_t{"sclass"}.sclass(strings_t{"cate0", "cate1", "cate2"}));
    check_stream(feature_t{"mclass"}.sclass(strings_t{"cate0", "cate1", "cate2", "cate3"}));
}

UTEST_END_MODULE()
