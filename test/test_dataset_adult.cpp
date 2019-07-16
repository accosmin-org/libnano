#include <utest/utest.h>
#include <nano/dataset.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_adult)

UTEST_CASE(config)
{
    auto dataset = dataset_t::all().get("adult");
    UTEST_REQUIRE(dataset);

    UTEST_CHECK_NOTHROW(dataset->config());

    json_t json;
    json["folds"] = 0;
    UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);

    json["folds"] = 101;
    UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);

    json["folds"] = 10;
    json["train_per"] = 9;
    UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);

    json["folds"] = 10;
    json["train_per"] = 91;
    UTEST_CHECK_THROW(dataset->config(json), std::invalid_argument);
}

UTEST_CASE(load)
{
    auto dataset = dataset_t::all().get("adult");
    UTEST_REQUIRE(dataset);

    json_t json;
    json["folds"] = 3;
    json["train_per"] = 60;
    UTEST_CHECK_NOTHROW(dataset->config(json));

    UTEST_REQUIRE(dataset->load());
    UTEST_CHECK_EQUAL(dataset->folds(), 3);
    UTEST_CHECK_EQUAL(dataset->ifeatures(), 14);
    UTEST_CHECK(!dataset->ifeature(0).discrete() && !dataset->ifeature(0).optional());
    UTEST_CHECK(dataset->ifeature(1).discrete() && dataset->ifeature(1).optional());
    UTEST_CHECK(!dataset->ifeature(2).discrete() && !dataset->ifeature(2).optional());
    UTEST_CHECK(dataset->ifeature(3).discrete() && !dataset->ifeature(3).optional());
    UTEST_CHECK(!dataset->ifeature(4).discrete() && !dataset->ifeature(4).optional());
    UTEST_CHECK(dataset->ifeature(5).discrete() && !dataset->ifeature(5).optional());
    UTEST_CHECK(dataset->ifeature(6).discrete() && dataset->ifeature(6).optional());
    UTEST_CHECK(dataset->ifeature(7).discrete() && !dataset->ifeature(7).optional());
    UTEST_CHECK(dataset->ifeature(8).discrete() && !dataset->ifeature(8).optional());
    UTEST_CHECK(dataset->ifeature(9).discrete() && !dataset->ifeature(9).optional());
    UTEST_CHECK(!dataset->ifeature(10).discrete() && !dataset->ifeature(10).optional());
    UTEST_CHECK(!dataset->ifeature(11).discrete() && !dataset->ifeature(11).optional());
    UTEST_CHECK(!dataset->ifeature(12).discrete() && !dataset->ifeature(12).optional());
    UTEST_CHECK(dataset->ifeature(13).discrete() && dataset->ifeature(13).optional());
    UTEST_CHECK(dataset->tfeature().discrete() && !dataset->tfeature().optional());

    for (size_t f = 0, folds = dataset->folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset->inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset->inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset->inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset->targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset->targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset->targets(fold_t{f, protocol::test});

        const auto tr_size = 60 * 32561 / 100;
        const auto vd_size = 32561 - tr_size;
        const auto te_size = 16281;

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(tr_size, 14, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(vd_size, 14, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(te_size, 14, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(tr_size, 2, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(vd_size, 2, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(te_size, 2, 1, 1));
    }
}

UTEST_END_MODULE()
