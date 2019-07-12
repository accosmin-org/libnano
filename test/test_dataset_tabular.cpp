#include <fstream>
#include <utest/utest.h>
#include <nano/dataset.h>

using namespace nano;

struct CSVFixture
{
    static auto data_path() { return "test_dataset_tabular_data.csv"; }
    static auto test_path() { return "test_dataset_tabular_test.csv"; }

    CSVFixture()
    {
        std::remove(data_path());
        std::remove(test_path());

        write_data(data_path());
        write_test(test_path());
    }

    ~CSVFixture()
    {
        std::remove(data_path());
        std::remove(test_path());
    }

    static void write_data(const char* path)
    {
        std::ofstream os(path);
        write(os, 1, 20);
        UTEST_REQUIRE(os);
    }

    static void write_test(const char* path)
    {
        std::ofstream os(path);
        write(os, 21, 10);
        UTEST_REQUIRE(os);
    }

    static void write(std::ostream& os, const int begin, const int size)
    {
        for (auto index = begin; index < begin + size; ++ index)
        {
            os << index << ",";
            (index % 2 == 0) ? (os << "?,") : (os << (3.0 - 0.2 * index) << ",");
            os << "cate" << (index % 3) << ",";
            (index % 4 == 0) ? (os << "?,") : (os << "cate_opt" << (index % 2) << ",");
            os << "\n";

            if (index % 7 == 0) { os << "\n"; }
            if (index % 9 == 0) { os << "#\n"; }
        }
    }
};

static auto make_feature_cont()
{
    const auto feature = feature_t::make_scalar("cont");

    UTEST_CHECK(feature_t::missing(feature_t::placeholder_value()));
    UTEST_CHECK(!feature_t::missing(0));

    UTEST_CHECK(!feature.discrete());
    UTEST_CHECK(!feature.optional());
    UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
    UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);

    return feature;
}

static auto make_feature_cont_opt()
{
    const auto feature = feature_t::make_scalar("cont_opt", "?");

    UTEST_CHECK(!feature.discrete());
    UTEST_CHECK(feature.optional());
    UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
    UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);

    return feature;
}

static auto make_feature_cate()
{
    const auto feature = feature_t::make_discrete("cate", {"cate0", "cate1", "cate2"});

    UTEST_CHECK(feature.discrete());
    UTEST_CHECK(!feature.optional());
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "cate2");
    UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
    UTEST_CHECK_THROW(feature.label(+3), std::out_of_range);
    UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());

    return feature;
}

static auto make_feature_cate_opt()
{
    const auto feature = feature_t::make_discrete("cate", {"cate_opt0", "cate_opt1"}, "?");

    UTEST_CHECK(feature.discrete());
    UTEST_CHECK(feature.optional());
    UTEST_CHECK_EQUAL(feature.label(0), "cate_opt0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate_opt1");
    UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
    UTEST_CHECK_THROW(feature.label(+2), std::out_of_range);
    UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());

    return feature;
}

UTEST_BEGIN_MODULE(test_dataset_tabular)

UTEST_CASE(empty)
{
    auto dataset = dataset_t::all().get("tabular");
    UTEST_REQUIRE(dataset);

    UTEST_CHECK_EQUAL(dataset->ifeatures(), 0);
    UTEST_CHECK_THROW(dataset->ifeature(0), std::out_of_range);
    UTEST_CHECK_THROW(dataset->tfeature(), std::out_of_range);
}

UTEST_CASE(config_no_target)
{
    auto dataset = dataset_t::all().get("tabular");
    UTEST_REQUIRE(dataset);

    const auto feature_cont = make_feature_cont();
    const auto feature_cont_opt = make_feature_cont_opt();
    const auto feature_cate = make_feature_cate();
    const auto feature_cate_opt = make_feature_cate_opt();

    json_t json;
    json["folds"] = 7;
    json["delim"] = ",";
    json["paths"] = {"path1", "path2"};
    json["features"] = {feature_cont.config(), feature_cont_opt.config(), feature_cate.config(), feature_cate_opt.config()};
    UTEST_CHECK_NOTHROW(dataset->config(json));

    UTEST_CHECK_EQUAL(dataset->folds(), 7);
    UTEST_CHECK_EQUAL(dataset->ifeatures(), 4);
    UTEST_CHECK_EQUAL(dataset->ifeature(0), feature_cont);
    UTEST_CHECK_EQUAL(dataset->ifeature(1), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset->ifeature(2), feature_cate);
    UTEST_CHECK_EQUAL(dataset->ifeature(3), feature_cate_opt);
    UTEST_CHECK_THROW(dataset->tfeature(), std::out_of_range);
}

UTEST_CASE(config_with_target)
{
    auto dataset = dataset_t::all().get("tabular");
    UTEST_REQUIRE(dataset);

    const auto feature_cont = make_feature_cont();
    const auto feature_cont_opt = make_feature_cont_opt();
    const auto feature_cate = make_feature_cate();
    const auto feature_cate_opt = make_feature_cate_opt();

    json_t json;
    json["folds"] = 7;
    json["delim"] = ",";
    json["paths"] = {"path1", "path2"};
    json["target"] = feature_cont.name();
    json["features"] = {feature_cont.config(), feature_cont_opt.config(), feature_cate.config(), feature_cate_opt.config()};
    UTEST_CHECK_NOTHROW(dataset->config(json));

    UTEST_CHECK_EQUAL(dataset->folds(), 7);
    UTEST_CHECK_EQUAL(dataset->ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset->ifeature(0), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset->ifeature(1), feature_cate);
    UTEST_CHECK_EQUAL(dataset->ifeature(2), feature_cate_opt);
    UTEST_CHECK_EQUAL(dataset->tfeature(), feature_cont);
}

UTEST_CASE(config_missing_attributes)
{
    auto dataset = dataset_t::all().get("tabular");
    UTEST_REQUIRE(dataset);

    const auto feature_cont = make_feature_cont();
    const auto feature_cont_opt = make_feature_cont_opt();
    const auto feature_cate = make_feature_cate();
    const auto feature_cate_opt = make_feature_cate_opt();

    json_t json;
    json["folds"] = 7;
    json["delim"] = ",";
    json["paths"] = {"path1", "path2"};
    json["features"] = {feature_cont.config(), feature_cont_opt.config()};

    {
        json.erase("folds");
        UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);
        json["folds"] = 7;
    }
    {
        json.erase("delim");
        UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);
        json["delim"] = ":";
    }
    {
        json.erase("paths");
        UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);
        json["paths"] = {"path"};
    }
    {
        json.erase("features");
        UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);
    }
    {
        json["features"] = {};
        UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);
    }
    {
        json["features"] = {feature_cont.config(), feature_cont_opt.config()};
        UTEST_CHECK_NOTHROW(dataset->config(json));
    }
}

UTEST_CASE(load_no_target)
{
    const auto fixture = CSVFixture{};

    auto dataset = dataset_t::all().get("tabular");
    UTEST_REQUIRE(dataset);

    const auto feature_cont = make_feature_cont();
    const auto feature_cont_opt = make_feature_cont_opt();
    const auto feature_cate = make_feature_cate();
    const auto feature_cate_opt = make_feature_cate_opt();

    json_t json;
    json["folds"] = 3;
    json["delim"] = ",";
    json["train_per"] = 60;
    json["paths"] = {fixture.data_path(), fixture.test_path()};
    json["features"] = {feature_cont.config(), feature_cont_opt.config(), feature_cate.config(), feature_cate_opt.config()};
    UTEST_REQUIRE_NOTHROW(dataset->config(json));

    UTEST_REQUIRE(dataset->load());
    UTEST_CHECK_EQUAL(dataset->folds(), 3);
    UTEST_CHECK_EQUAL(dataset->ifeatures(), 4);
    UTEST_CHECK_EQUAL(dataset->ifeature(0), feature_cont);
    UTEST_CHECK_EQUAL(dataset->ifeature(1), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset->ifeature(2), feature_cate);
    UTEST_CHECK_EQUAL(dataset->ifeature(3), feature_cate_opt);
    UTEST_CHECK_THROW(dataset->tfeature(), std::out_of_range);

    for (size_t f = 0, folds = dataset->folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset->inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset->inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset->inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset->targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset->targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset->targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(18, 4, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(6, 4, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(6, 4, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(18, 0, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(6, 0, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(6, 0, 1, 1));
    }
}

UTEST_CASE(load_with_cont_target)
{
    const auto fixture = CSVFixture{};

    auto dataset = dataset_t::all().get("tabular");
    UTEST_REQUIRE(dataset);

    const auto feature_cont = make_feature_cont();
    const auto feature_cont_opt = make_feature_cont_opt();
    const auto feature_cate = make_feature_cate();
    const auto feature_cate_opt = make_feature_cate_opt();

    json_t json;
    json["folds"] = 2;
    json["delim"] = ",";
    json["target"] = feature_cont.name();
    json["paths"] = {fixture.data_path(), fixture.test_path()};
    json["features"] = {feature_cont.config(), feature_cont_opt.config(), feature_cate.config(), feature_cate_opt.config()};
    UTEST_REQUIRE_NOTHROW(dataset->config(json));

    UTEST_REQUIRE(dataset->load());
    UTEST_CHECK_EQUAL(dataset->folds(), 2);
    UTEST_CHECK_EQUAL(dataset->ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset->ifeature(0), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset->ifeature(1), feature_cate);
    UTEST_CHECK_EQUAL(dataset->ifeature(2), feature_cate_opt);
    UTEST_CHECK_EQUAL(dataset->tfeature(), feature_cont);

    for (size_t f = 0, folds = dataset->folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset->inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset->inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset->inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset->targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset->targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset->targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(24, 3, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(3, 3, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(3, 3, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(24, 1, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(3, 1, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(3, 1, 1, 1));
    }
}

UTEST_CASE(load_with_cate_target)
{
    const auto fixture = CSVFixture{};

    auto dataset = dataset_t::all().get("tabular");
    UTEST_REQUIRE(dataset);

    const auto feature_cont = make_feature_cont();
    const auto feature_cont_opt = make_feature_cont_opt();
    const auto feature_cate = make_feature_cate();
    const auto feature_cate_opt = make_feature_cate_opt();

    json_t json;
    json["folds"] = 7;
    json["delim"] = ",";
    json["target"] = feature_cate.name();
    json["paths"] = {fixture.data_path(), fixture.test_path()};
    json["features"] = {feature_cont.config(), feature_cont_opt.config(), feature_cate.config(), feature_cate_opt.config()};
    UTEST_REQUIRE_NOTHROW(dataset->config(json));

    UTEST_REQUIRE(dataset->load());
    UTEST_CHECK_EQUAL(dataset->folds(), 7);
    UTEST_CHECK_EQUAL(dataset->ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset->ifeature(0), feature_cont);
    UTEST_CHECK_EQUAL(dataset->ifeature(1), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset->ifeature(2), feature_cate_opt);
    UTEST_CHECK_EQUAL(dataset->tfeature(), feature_cate);

    for (size_t f = 0, folds = dataset->folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset->inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset->inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset->inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset->targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset->targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset->targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(24, 3, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(3, 3, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(3, 3, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(24, 3, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(3, 3, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(3, 3, 1, 1));
    }
}

UTEST_END_MODULE()
