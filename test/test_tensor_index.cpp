#include <nano/string.h>
#include <utest/utest.h>
#include <nano/tensor/index.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_tensor_index)

UTEST_CASE(dims1d)
{
    const auto dims1 = nano::make_dims(3);
    const auto dims2 = nano::make_dims(3);
    const auto dims3 = nano::make_dims(1);

    UTEST_CHECK_EQUAL(dims1, dims1);
    UTEST_CHECK_EQUAL(dims1, dims2);
    UTEST_CHECK_NOT_EQUAL(dims2, dims3);

    UTEST_CHECK_EQUAL(scat(dims1), "3");
    UTEST_CHECK_EQUAL(scat(dims2), "3");
    UTEST_CHECK_EQUAL(scat(dims3), "1");
}

UTEST_CASE(dims2d)
{
    const auto dims1 = nano::make_dims(3, 7);
    const auto dims2 = nano::make_dims(7, 5);
    const auto dims3 = nano::make_dims(3, 7);

    UTEST_CHECK_EQUAL(dims1, dims1);
    UTEST_CHECK_EQUAL(dims1, dims3);
    UTEST_CHECK_NOT_EQUAL(dims1, dims2);

    UTEST_CHECK_EQUAL(scat(dims1), "3x7");
    UTEST_CHECK_EQUAL(scat(dims2), "7x5");
    UTEST_CHECK_EQUAL(scat(dims3), "3x7");
}

UTEST_CASE(dims3d)
{
    const auto dims1 = nano::make_dims(3, 7, 5);
    const auto dims2 = nano::make_dims(7, 5, 3);
    const auto dims3 = nano::make_dims(1, 1, 1);

    UTEST_CHECK_EQUAL(dims1, dims1);
    UTEST_CHECK_NOT_EQUAL(dims2, dims1);
    UTEST_CHECK_NOT_EQUAL(dims2, dims3);

    UTEST_CHECK_EQUAL(scat(dims1), "3x7x5");
    UTEST_CHECK_EQUAL(scat(dims2), "7x5x3");
    UTEST_CHECK_EQUAL(scat(dims3), "1x1x1");
}

UTEST_CASE(index1d)
{
    const auto dims = nano::make_dims(7);

    UTEST_CHECK_EQUAL(std::get<0>(dims), 7);
    UTEST_CHECK_EQUAL(nano::size(dims), 7);

    UTEST_CHECK_EQUAL(nano::index(dims, 0), 0);
    UTEST_CHECK_EQUAL(nano::index(dims, 1), 1);
    UTEST_CHECK_EQUAL(nano::index(dims, 6), 6);

    UTEST_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0));
    UTEST_CHECK_EQUAL(nano::index0(dims, 6), nano::index(dims, 6));

    UTEST_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(7));
}

UTEST_CASE(index2d)
{
    const auto dims = nano::make_dims(7, 5);

    UTEST_CHECK_EQUAL(dims, nano::cat_dims(7, nano::make_dims(5)));

    UTEST_CHECK_EQUAL(std::get<0>(dims), 7);
    UTEST_CHECK_EQUAL(std::get<1>(dims), 5);
    UTEST_CHECK_EQUAL(nano::size(dims), 35);

    UTEST_CHECK_EQUAL(nano::index(dims, 0, 1), 1);
    UTEST_CHECK_EQUAL(nano::index(dims, 0, 4), 4);
    UTEST_CHECK_EQUAL(nano::index(dims, 1, 0), 5);
    UTEST_CHECK_EQUAL(nano::index(dims, 3, 2), 17);
    UTEST_CHECK_EQUAL(nano::index(dims, 6, 4), 34);

    UTEST_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0, 0));
    UTEST_CHECK_EQUAL(nano::index0(dims, 3), nano::index(dims, 3, 0));
    UTEST_CHECK_EQUAL(nano::index0(dims, 3, 1), nano::index(dims, 3, 1));

    UTEST_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(7, 5));
    UTEST_CHECK_EQUAL(nano::dims0(dims, 3), nano::make_dims(5));
}

UTEST_CASE(index3d)
{
    const auto dims = nano::make_dims(3, 7, 5);

    UTEST_CHECK_EQUAL(dims, nano::cat_dims(3, nano::make_dims(7, 5)));

    UTEST_CHECK_EQUAL(std::get<0>(dims), 3);
    UTEST_CHECK_EQUAL(std::get<1>(dims), 7);
    UTEST_CHECK_EQUAL(std::get<2>(dims), 5);
    UTEST_CHECK_EQUAL(nano::size(dims), 105);

    UTEST_CHECK_EQUAL(nano::index(dims, 0, 0, 1), 1);
    UTEST_CHECK_EQUAL(nano::index(dims, 0, 0, 4), 4);
    UTEST_CHECK_EQUAL(nano::index(dims, 0, 1, 0), 5);
    UTEST_CHECK_EQUAL(nano::index(dims, 0, 2, 1), 11);
    UTEST_CHECK_EQUAL(nano::index(dims, 1, 2, 1), 46);
    UTEST_CHECK_EQUAL(nano::index(dims, 1, 0, 3), 38);
    UTEST_CHECK_EQUAL(nano::index(dims, 2, 4, 1), 91);
    UTEST_CHECK_EQUAL(nano::index(dims, 2, 6, 4), 104);

    UTEST_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0, 0, 0));
    UTEST_CHECK_EQUAL(nano::index0(dims, 2), nano::index(dims, 2, 0, 0));
    UTEST_CHECK_EQUAL(nano::index0(dims, 2, 4), nano::index(dims, 2, 4, 0));
    UTEST_CHECK_EQUAL(nano::index0(dims, 2, 4, 3), nano::index(dims, 2, 4, 3));

    UTEST_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(3, 7, 5));
    UTEST_CHECK_EQUAL(nano::dims0(dims, 2), nano::make_dims(7, 5));
    UTEST_CHECK_EQUAL(nano::dims0(dims, 2, 4), nano::make_dims(5));
}

UTEST_CASE(range)
{
    const auto range_def = nano::tensor_range_t{};
    const auto range_ok0 = nano::make_range(0, 1);
    const auto range_ok1 = nano::make_range(1, 3);
    const auto range_nok0 = nano::make_range(-1, 1);
    const auto range_nok1 = nano::make_range(+3, 1);

    UTEST_CHECK_EQUAL(range_def.begin(), 0);
    UTEST_CHECK_EQUAL(range_ok0.begin(), 0);
    UTEST_CHECK_EQUAL(range_ok1.begin(), 1);
    UTEST_CHECK_EQUAL(range_nok0.begin(), -1);
    UTEST_CHECK_EQUAL(range_nok1.begin(), +3);

    UTEST_CHECK_EQUAL(range_def.end(), 0);
    UTEST_CHECK_EQUAL(range_ok0.end(), 1);
    UTEST_CHECK_EQUAL(range_ok1.end(), 3);
    UTEST_CHECK_EQUAL(range_nok0.end(), 1);
    UTEST_CHECK_EQUAL(range_nok1.end(), 1);

    UTEST_CHECK_EQUAL(range_def.size(), 0);
    UTEST_CHECK_EQUAL(range_ok0.size(), 1);
    UTEST_CHECK_EQUAL(range_ok1.size(), 2);
    UTEST_CHECK_EQUAL(range_nok0.size(), 2);
    UTEST_CHECK_EQUAL(range_nok1.size(), -2);

    UTEST_CHECK(!range_def.valid(-1));
    UTEST_CHECK(!range_def.valid(+0));
    UTEST_CHECK(!range_def.valid(+1));

    UTEST_CHECK(!range_ok0.valid(-2));
    UTEST_CHECK(!range_ok0.valid(-1));
    UTEST_CHECK(!range_ok0.valid(+0));
    UTEST_CHECK(range_ok0.valid(+1));
    UTEST_CHECK(range_ok0.valid(+2));
    UTEST_CHECK(range_ok0.valid(+3));

    UTEST_CHECK(!range_ok1.valid(-2));
    UTEST_CHECK(!range_ok1.valid(-1));
    UTEST_CHECK(!range_ok1.valid(+0));
    UTEST_CHECK(!range_ok1.valid(+1));
    UTEST_CHECK(!range_ok1.valid(+2));
    UTEST_CHECK(range_ok1.valid(+3));
    UTEST_CHECK(range_ok1.valid(+4));

    UTEST_CHECK(!range_nok0.valid(-2));
    UTEST_CHECK(!range_nok0.valid(-1));
    UTEST_CHECK(!range_nok0.valid(+0));
    UTEST_CHECK(!range_nok0.valid(+1));
    UTEST_CHECK(!range_nok0.valid(+2));
    UTEST_CHECK(!range_nok0.valid(+3));

    UTEST_CHECK(!range_nok1.valid(-2));
    UTEST_CHECK(!range_nok1.valid(-1));
    UTEST_CHECK(!range_nok1.valid(+0));
    UTEST_CHECK(!range_nok1.valid(+1));
    UTEST_CHECK(!range_nok1.valid(+2));
    UTEST_CHECK(!range_nok1.valid(+3));
    UTEST_CHECK(!range_nok1.valid(+4));
}

UTEST_END_MODULE()
