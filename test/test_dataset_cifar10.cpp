#include <utest/utest.h>
#include <nano/dataset/imclass.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_cifar10)

UTEST_CASE(load)
{
    auto dataset = imclass_dataset_t::all().get("cifar10");

    UTEST_REQUIRE(dataset);
    UTEST_CHECK_NOTHROW(dataset->folds(1));
    UTEST_CHECK_NOTHROW(dataset->train_percentage(80));

    UTEST_REQUIRE(dataset->load());
    UTEST_CHECK_EQUAL(dataset->folds(), 1);
    UTEST_CHECK(dataset->tfeature().discrete());
    UTEST_CHECK(!dataset->tfeature().optional());
    UTEST_CHECK_EQUAL(dataset->idim(), make_dims(32, 32, 3));
    UTEST_CHECK_EQUAL(dataset->tdim(), make_dims(10, 1, 1));
    UTEST_CHECK_EQUAL(dataset->tfeature().labels().size(), 10U);

    for (size_t f = 0, folds = dataset->folds(); f < folds; ++ f)
    {
        const auto tr_size = 80 * 50000 / 100;
        const auto vd_size = 50000 - tr_size;
        const auto te_size = 10000;

        UTEST_CHECK_EQUAL(dataset->samples(fold_t{f, protocol::train}), tr_size);
        UTEST_CHECK_EQUAL(dataset->samples(fold_t{f, protocol::valid}), vd_size);
        UTEST_CHECK_EQUAL(dataset->samples(fold_t{f, protocol::test}), te_size);
    }
}

UTEST_END_MODULE()
