#include <utest/utest.h>
#include <nano/dataset/tabular.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_iris)

UTEST_CASE(load)
{
    auto dataset = tabular_dataset_t::all().get("iris");
    UTEST_REQUIRE(dataset);

    UTEST_CHECK_NOTHROW(dataset->folds(3));
    UTEST_CHECK_NOTHROW(dataset->train_percentage(60));

    UTEST_REQUIRE(dataset->load());
    UTEST_CHECK_EQUAL(dataset->folds(), 3);
    UTEST_CHECK_EQUAL(dataset->ifeatures(), 4);
    UTEST_CHECK(dataset->tfeature().discrete() && !dataset->tfeature().optional());
    UTEST_CHECK(!dataset->ifeature(0).discrete() && !dataset->ifeature(0).optional());
    UTEST_CHECK(!dataset->ifeature(1).discrete() && !dataset->ifeature(1).optional());
    UTEST_CHECK(!dataset->ifeature(2).discrete() && !dataset->ifeature(2).optional());
    UTEST_CHECK(!dataset->ifeature(3).discrete() && !dataset->ifeature(3).optional());

    for (size_t f = 0, folds = dataset->folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset->inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset->inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset->inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset->targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset->targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset->targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(90, 4, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(30, 4, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(30, 4, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(90, 3, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(30, 3, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(30, 3, 1, 1));
    }
}

UTEST_END_MODULE()
