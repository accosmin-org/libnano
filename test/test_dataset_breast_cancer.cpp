#include <utest/utest.h>
#include <nano/dataset/tabular.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_breast_cancer)

UTEST_CASE(load)
{
    const auto dataset = tabular_dataset_t::all().get("breast-cancer");

    UTEST_REQUIRE(dataset);
    UTEST_REQUIRE_NOTHROW(dataset->load());

    UTEST_CHECK_EQUAL(dataset->features(), 31);
    UTEST_CHECK(dataset->target().discrete() && !dataset->target().optional());
    UTEST_CHECK(!dataset->feature(0).discrete() && !dataset->feature(0).optional());
    UTEST_CHECK(!dataset->feature(1).discrete() && !dataset->feature(1).optional());
    UTEST_CHECK(!dataset->feature(2).discrete() && !dataset->feature(2).optional());
    UTEST_CHECK(!dataset->feature(3).discrete() && !dataset->feature(3).optional());
    UTEST_CHECK(!dataset->feature(4).discrete() && !dataset->feature(4).optional());
    UTEST_CHECK(!dataset->feature(5).discrete() && !dataset->feature(5).optional());
    UTEST_CHECK(!dataset->feature(6).discrete() && !dataset->feature(6).optional());
    UTEST_CHECK(!dataset->feature(7).discrete() && !dataset->feature(7).optional());
    UTEST_CHECK(!dataset->feature(8).discrete() && !dataset->feature(8).optional());
    UTEST_CHECK(!dataset->feature(9).discrete() && !dataset->feature(9).optional());
    UTEST_CHECK(!dataset->feature(10).discrete() && !dataset->feature(10).optional());
    UTEST_CHECK(!dataset->feature(11).discrete() && !dataset->feature(11).optional());
    UTEST_CHECK(!dataset->feature(12).discrete() && !dataset->feature(12).optional());
    UTEST_CHECK(!dataset->feature(13).discrete() && !dataset->feature(13).optional());
    UTEST_CHECK(!dataset->feature(14).discrete() && !dataset->feature(14).optional());
    UTEST_CHECK(!dataset->feature(15).discrete() && !dataset->feature(15).optional());
    UTEST_CHECK(!dataset->feature(16).discrete() && !dataset->feature(16).optional());
    UTEST_CHECK(!dataset->feature(17).discrete() && !dataset->feature(17).optional());
    UTEST_CHECK(!dataset->feature(18).discrete() && !dataset->feature(18).optional());
    UTEST_CHECK(!dataset->feature(19).discrete() && !dataset->feature(19).optional());
    UTEST_CHECK(!dataset->feature(20).discrete() && !dataset->feature(20).optional());
    UTEST_CHECK(!dataset->feature(21).discrete() && !dataset->feature(21).optional());
    UTEST_CHECK(!dataset->feature(22).discrete() && !dataset->feature(22).optional());
    UTEST_CHECK(!dataset->feature(23).discrete() && !dataset->feature(23).optional());
    UTEST_CHECK(!dataset->feature(24).discrete() && !dataset->feature(24).optional());
    UTEST_CHECK(!dataset->feature(25).discrete() && !dataset->feature(25).optional());
    UTEST_CHECK(!dataset->feature(26).discrete() && !dataset->feature(26).optional());
    UTEST_CHECK(!dataset->feature(27).discrete() && !dataset->feature(27).optional());
    UTEST_CHECK(!dataset->feature(28).discrete() && !dataset->feature(28).optional());
    UTEST_CHECK(!dataset->feature(29).discrete() && !dataset->feature(29).optional());
    UTEST_CHECK(!dataset->feature(30).discrete() && !dataset->feature(30).optional());

    UTEST_CHECK_EQUAL(dataset->idim(), make_dims(31, 1, 1));
    UTEST_CHECK_EQUAL(dataset->tdim(), make_dims(2, 1, 1));

    UTEST_CHECK_EQUAL(dataset->samples(), 569);
    UTEST_CHECK_EQUAL(dataset->test_samples(), arange(0, 0));
    UTEST_CHECK_EQUAL(dataset->train_samples(), arange(0, 569));

    UTEST_CHECK_EQUAL(dataset->type(), task_type::sclassification);
}

UTEST_END_MODULE()
