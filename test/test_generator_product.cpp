#include <utest/utest.h>
#include "fixture/generator.h"
#include "fixture/generator_dataset.h"
#include <nano/generator/pairwise_product.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_generator_product)

UTEST_CASE(empty)
{
    const auto dataset = make_dataset(10, string_t::npos);
    const auto generator = dataset_generator_t{dataset};

    UTEST_CHECK_EQUAL(generator.columns(), 0);
    UTEST_CHECK_EQUAL(generator.features(), 0);
}

UTEST_CASE(product)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    UTEST_CHECK_NOTHROW(generator.add<pairwise_generator_t<pairwise_product_t>>());

    UTEST_REQUIRE_EQUAL(generator.features(), 6);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"product(scalar0,scalar0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"product(scalar0,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"product(scalar0,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"product(scalar1,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"product(scalar1,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"product(scalar2,scalar2)"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(10), 1, 0, 1, 4, 9, 16, 25, 36, 49, 64));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(10), 2, N, 0, N, 6, Na, 20, Na, 42, Na));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(10), 3, N, N, 0, N, Na, 15, Na, Na, 48));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(10), 4, N, 0, N, 4, Na, 16, Na, 36, Na));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(10), 6, N, N, N, N, Na, 12, Na, Na, Na));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(10), 9, N, N, 0, N, Na, +9, Na, Na, 36));

    check_flatten(generator, make_tensor<scalar_t>(make_dims(10, 6),
        1, 2, 3, 4, 6, 9,
        0, N, N, N, N, N,
        1, 0, N, 0, N, N,
        4, N, 0, N, N, 0,
        9, 6, N, 4, N, N,
        16, Na, Na, Na, Na, Na,
        25, 20, 15, 16, 12, +9,
        36, Na, Na, Na, Na, Na,
        49, 42, Na, 36, Na, Na,
        64, Na, 48, Na, Na, 36),
        make_indices(0, 1, 2, 3, 4, 5));
}

UTEST_END_MODULE()
