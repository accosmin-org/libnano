#include <utest/utest.h>
#include <nano/mlearn/feature.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_mlearn_feature)

UTEST_CASE(_default)
{
    feature_t feature;
    UTEST_CHECK_EQUAL(static_cast<bool>(feature), false);

    feature = feature_t{"feature"};
    UTEST_CHECK_EQUAL(static_cast<bool>(feature), true);

    UTEST_CHECK(feature_t::missing(feature_t::placeholder_value()));
    UTEST_CHECK(!feature_t::missing(0));
}

UTEST_CASE(discrete)
{
    auto feature = feature_t{"cate"};
    UTEST_CHECK(!feature.discrete());

    feature.labels(4);
    UTEST_CHECK(feature.discrete());
    UTEST_CHECK_EQUAL(feature.label(0), "");
    UTEST_CHECK_EQUAL(feature.label(1), "");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label(""), string_t::npos);
    UTEST_CHECK_EQUAL(feature.label(0), "");
    UTEST_CHECK_EQUAL(feature.label(1), "");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate0"), 0U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate1"), 1U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate1"), 1U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate2"), 2U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "cate2");
    UTEST_CHECK_EQUAL(feature.label(3), "");

    UTEST_CHECK_EQUAL(feature.set_label("cate3"), 3U);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "cate2");
    UTEST_CHECK_EQUAL(feature.label(3), "cate3");

    UTEST_CHECK_EQUAL(feature.set_label("cate4"), string_t::npos);
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "cate2");
    UTEST_CHECK_EQUAL(feature.label(3), "cate3");
}

UTEST_CASE(compare)
{
    const auto make_feature_cont = [] (const string_t& name)
    {
        auto feature = feature_t{name};
        UTEST_CHECK(!feature.discrete());
        UTEST_CHECK(!feature.optional());
        UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
        UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);
        return feature;
    };

    const auto make_feature_cont_opt = [] (const string_t& name)
    {
        auto feature = feature_t{name}.placeholder("?");
        UTEST_CHECK(!feature.discrete());
        UTEST_CHECK(feature.optional());
        UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
        UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);
        return feature;
    };

    const auto make_feature_cate = [] (const string_t& name)
    {
        auto feature = feature_t{name}.labels({"cate0", "cate1", "cate2"});
        UTEST_CHECK(feature.discrete());
        UTEST_CHECK(!feature.optional());
        UTEST_CHECK_EQUAL(feature.label(0), "cate0");
        UTEST_CHECK_EQUAL(feature.label(1), "cate1");
        UTEST_CHECK_EQUAL(feature.label(2), "cate2");
        UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
        UTEST_CHECK_THROW(feature.label(+3), std::out_of_range);
        UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());
        return feature;
    };

    const auto make_feature_cate_opt = [] (const string_t& name)
    {
        auto feature = feature_t{name}.labels({"cate_opt0", "cate_opt1"}).placeholder("?");
        UTEST_CHECK(feature.discrete());
        UTEST_CHECK(feature.optional());
        UTEST_CHECK_EQUAL(feature.label(0), "cate_opt0");
        UTEST_CHECK_EQUAL(feature.label(1), "cate_opt1");
        UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
        UTEST_CHECK_THROW(feature.label(+2), std::out_of_range);
        UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());
        return feature;
    };

    const auto to_string = [] (const feature_t& feature)
    {
        std::stringstream stream;
        stream << feature;
        return stream.str();
    };

    UTEST_CHECK_EQUAL(make_feature_cont("f"), make_feature_cont("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont("gf"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont("f")), "name=f,labels[],placeholder=");

    UTEST_CHECK_EQUAL(make_feature_cont_opt("f"), make_feature_cont_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont_opt("f"), make_feature_cont_opt("ff"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cont_opt("f")), "name=f,labels[],placeholder=?");

    UTEST_CHECK_EQUAL(make_feature_cate("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate("f"), make_feature_cate("x"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate("f")), "name=f,labels[cate0,cate1,cate2],placeholder=");

    UTEST_CHECK_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cate_opt("f"), make_feature_cate_opt("x"));
    UTEST_CHECK_EQUAL(to_string(make_feature_cate_opt("f")), "name=f,labels[cate_opt0,cate_opt1],placeholder=?");

    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cate("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cont_opt("f"));
    UTEST_CHECK_NOT_EQUAL(make_feature_cont("f"), make_feature_cate_opt("f"));
}

UTEST_END_MODULE()
