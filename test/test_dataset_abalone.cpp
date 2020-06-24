#include <utest/utest.h>
#include <nano/dataset/tabular.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_abalone)

UTEST_CASE(load)
{
    auto dataset = tabular_dataset_t::all().get("abalone");

    UTEST_REQUIRE(dataset);
    UTEST_CHECK_NOTHROW(dataset->folds(1));
    UTEST_CHECK_NOTHROW(dataset->train_percentage(60));

    UTEST_REQUIRE(dataset->load());
    UTEST_CHECK_EQUAL(dataset->folds(), 1);
    UTEST_CHECK_EQUAL(dataset->ifeatures(), 8);
    UTEST_CHECK(dataset->tfeature().discrete() && !dataset->tfeature().optional());
    UTEST_CHECK(dataset->ifeature(0).discrete() && !dataset->ifeature(0).optional());
    UTEST_CHECK(!dataset->ifeature(1).discrete() && !dataset->ifeature(1).optional());
    UTEST_CHECK(!dataset->ifeature(2).discrete() && !dataset->ifeature(2).optional());
    UTEST_CHECK(!dataset->ifeature(3).discrete() && !dataset->ifeature(3).optional());
    UTEST_CHECK(!dataset->ifeature(4).discrete() && !dataset->ifeature(4).optional());
    UTEST_CHECK(!dataset->ifeature(5).discrete() && !dataset->ifeature(5).optional());
    UTEST_CHECK(!dataset->ifeature(6).discrete() && !dataset->ifeature(6).optional());
    UTEST_CHECK(!dataset->ifeature(7).discrete() && !dataset->ifeature(7).optional());

    for (size_t f = 0, folds = dataset->folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset->inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset->inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset->inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset->targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset->targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset->targets(fold_t{f, protocol::test});

        const auto tr_size = 60 * 3133 / 100;
        const auto vd_size = 3133 - tr_size;
        const auto te_size = 1044;

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(tr_size, 8, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(vd_size, 8, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(te_size, 8, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(tr_size, 29, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(vd_size, 29, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(te_size, 29, 1, 1));
    }
}

UTEST_END_MODULE()
