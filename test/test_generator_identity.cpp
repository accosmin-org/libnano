#include <utest/utest.h>
#include "fixture/generator.h"
#include "fixture/generator_dataset.h"
#include <nano/generator/elemwise_identity.h>

using namespace nano;

static auto make_generator(const dataset_t& dataset)
{
    auto generator = dataset_generator_t{dataset};
    add_generator<elemwise_generator_t<sclass_identity_t>>(generator);
    add_generator<elemwise_generator_t<mclass_identity_t>>(generator);
    add_generator<elemwise_generator_t<scalar_identity_t>>(generator);
    add_generator<elemwise_generator_t<struct_identity_t>>(generator);
    return generator;
}

static auto keep(const tensor2d_t& flatten, const indices_t& columns_to_keep)
{
    const auto [samples, columns] = flatten.dims();

    tensor2d_t tensor(samples, columns_to_keep.size());
    tensor_size_t column2 = 0;
    for (const auto column : columns_to_keep)
    {
        tensor.matrix().col(column2 ++) = flatten.matrix().col(column);
    }
    return tensor;
}

static auto remove(const tensor2d_t& flatten, const indices_t& columns_to_remove)
{
    const auto [samples, columns] = flatten.dims();

    const auto* const begin = ::nano::begin(columns_to_remove);
    const auto* const end = ::nano::end(columns_to_remove);

    tensor2d_t tensor(samples, columns - columns_to_remove.size());
    for (tensor_size_t column = 0, column2 = 0; column < columns; ++ column)
    {
        if (std::find(begin, end, column) == end)
        {
            tensor.matrix().col(column2 ++) = flatten.matrix().col(column);
        }
    }
    return tensor;
}

static auto drop(const tensor2d_t& flatten, const indices_t& columns)
{
    tensor2d_t tensor = flatten;
    for (const auto column : columns)
    {
        tensor.matrix().array().col(column) = NaN;
    }
    return tensor;
}

static auto expected_sclass0() { return make_features()[2]; }
static auto expected_sclass1() { return make_features()[3]; }
static auto expected_sclass2() { return make_features()[4]; }
static auto expected_mclass0() { return make_features()[0]; }
static auto expected_mclass1() { return make_features()[1]; }
static auto expected_scalar0() { return make_features()[5]; }
static auto expected_scalar1() { return make_features()[6]; }
static auto expected_scalar2() { return make_features()[7]; }
static auto expected_struct0() { return make_features()[8]; }
static auto expected_struct1() { return make_features()[9]; }
static auto expected_struct2() { return make_features()[10]; }

static auto expected_select_sclass0()
{
    return make_tensor<int32_t>(make_dims(10), +2, -1, +1, -1, +0, -1, +2, -1, +1, -1);
}

static auto expected_select_sclass1()
{
    return make_tensor<int32_t>(make_dims(10), +1, +0, +1, +0, +1, +0, +1, +0, +1, +0);
}

static auto expected_select_sclass2()
{
    return make_tensor<int32_t>(make_dims(10), +0, -1, +0, -1, +0, -1, +0, -1, +0, -1);
}

static auto expected_select_mclass0()
{
    return make_tensor<int8_t>(make_dims(10, 3),
        +0, +1, +1,
        +1, +0, +0,
        +0, +1, +0,
        +1, +0, +0,
        +0, +1, +0,
        +1, +0, +0,
        +0, +1, +1,
        +1, +0, +0,
        +0, +1, +0,
        +1, +0, +0);
}

static auto expected_select_mclass1()
{
    return make_tensor<int8_t>(make_dims(10, 4),
        +0, +1, +1, +0,
        -1, -1, -1, -1,
        +0, +1, +0, +0,
        -1, -1, -1, -1,
        +0, +1, +0, +0,
        -1, -1, -1, -1,
        +0, +1, +1, +0,
        -1, -1, -1, -1,
        +0, +1, +0, +0,
        -1, -1, -1, -1);
}

static auto expected_select_scalar0()
{
    return make_tensor<scalar_t>(make_dims(10), -1, +0, +1, +2, +3, +4, +5, +6, +7, +8);
}

static auto expected_select_scalar1()
{
    return make_tensor<scalar_t>(make_dims(10), -2, Na, +0, Na, +2, Na, +4, Na, +6, Na);
}

static auto expected_select_scalar2()
{
    return make_tensor<scalar_t>(make_dims(10), -3, Na, Na, +0, Na, Na, +3, Na, Na, +6);
}

static auto expected_select_struct0()
{
    return make_tensor<scalar_t>(make_dims(10, 1, 2, 2),
        +1, +0, +0, +0,
        +2, +1, +1, +1,
        +3, +2, +2, +2,
        +4, +3, +3, +3,
        +5, +4, +4, +4,
        +6, +5, +5, +5,
        +7, +6, +6, +6,
        +8, +7, +7, +7,
        +9, +8, +8, +8,
        +10, +9, +9, +9);
}

static auto expected_select_struct1()
{
    return make_tensor<scalar_t>(make_dims(10, 2, 1, 3),
        +1, +0, +0, +0, +0, +0,
        Na, Na, Na, Na, Na, Na,
        +3, +2, +2, +2, +2, +2,
        Na, Na, Na, Na, Na, Na,
        +5, +4, +4, +4, +4, +4,
        Na, Na, Na, Na, Na, Na,
        +7, +6, +6, +6, +6, +6,
        Na, Na, Na, Na, Na, Na,
        +9, +8, +8, +8, +8, +8,
        Na, Na, Na, Na, Na, Na);
}

static auto expected_select_struct2()
{
    return make_tensor<scalar_t>(make_dims(10, 3, 1, 1),
        +1, +0, +0,
        Na, Na, Na,
        Na, Na, Na,
        +4, +3, +3,
        Na, Na, Na,
        Na, Na, Na,
        +7, +6, +6,
        Na, Na, Na,
        Na, Na, Na,
        +10, +9, +9);
}

static auto expected_flatten()
{
    return make_tensor<scalar_t>(make_dims(10, 30),
        -1, -1, +1, -1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, -1, -2, -3, +1, +0, +0, +0, +1, +0, +0, +0, +0, +0, +1, +0, +0,
        Na, Na, Na, +1, -1, Na, Na, +1, -1, -1, Na, Na, Na, Na, +0, Na, Na, +2, +1, +1, +1, Na, Na, Na, Na, Na, Na, Na, Na, Na,
        -1, +1, -1, -1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1, +1, +0, Na, +3, +2, +2, +2, +3, +2, +2, +2, +2, +2, Na, Na, Na,
        Na, Na, Na, +1, -1, Na, Na, +1, -1, -1, Na, Na, Na, Na, +2, Na, +0, +4, +3, +3, +3, Na, Na, Na, Na, Na, Na, +4, +3, +3,
        +1, -1, -1, -1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1, +3, +2, Na, +5, +4, +4, +4, +5, +4, +4, +4, +4, +4, Na, Na, Na,
        Na, Na, Na, +1, -1, Na, Na, +1, -1, -1, Na, Na, Na, Na, +4, Na, Na, +6, +5, +5, +5, Na, Na, Na, Na, Na, Na, Na, Na, Na,
        -1, -1, +1, -1, +1, +1, -1, -1, +1, +1, -1, +1, +1, -1, +5, +4, +3, +7, +6, +6, +6, +7, +6, +6, +6, +6, +6, +7, +6, +6,
        Na, Na, Na, +1, -1, Na, Na, +1, -1, -1, Na, Na, Na, Na, +6, Na, Na, +8, +7, +7, +7, Na, Na, Na, Na, Na, Na, Na, Na, Na,
        -1, +1, -1, -1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1, +7, +6, Na, +9, +8, +8, +8, +9, +8, +8, +8, +8, +8, Na, Na, Na,
        Na, Na, Na, +1, -1, Na, Na, +1, -1, -1, Na, Na, Na, Na, +8, Na, +6, 10, +9, +9, +9, Na, Na, Na, Na, Na, Na, 10, +9, +9);
}

UTEST_BEGIN_MODULE(test_generator_identity)

UTEST_CASE(empty)
{
    const auto dataset = make_dataset(10, string_t::npos);
    const auto generator = dataset_generator_t{dataset};

    UTEST_CHECK_EQUAL(generator.columns(), 0);
    UTEST_CHECK_EQUAL(generator.features(), 0);
}

UTEST_CASE(unsupervised)
{
    const auto dataset = make_dataset(10, string_t::npos);
    const auto generator = make_generator(dataset);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::unsupervised);

    UTEST_REQUIRE_EQUAL(generator.features(), 11);
    UTEST_CHECK_EQUAL(generator.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(generator.feature(1), expected_sclass1());
    UTEST_CHECK_EQUAL(generator.feature(2), expected_sclass2());
    UTEST_CHECK_EQUAL(generator.feature(3), expected_mclass0());
    UTEST_CHECK_EQUAL(generator.feature(4), expected_mclass1());
    UTEST_CHECK_EQUAL(generator.feature(5), expected_scalar0());
    UTEST_CHECK_EQUAL(generator.feature(6), expected_scalar1());
    UTEST_CHECK_EQUAL(generator.feature(7), expected_scalar2());
    UTEST_CHECK_EQUAL(generator.feature(8), expected_struct0());
    UTEST_CHECK_EQUAL(generator.feature(9), expected_struct1());
    UTEST_CHECK_EQUAL(generator.feature(10), expected_struct2());

    check_select(generator, 0, expected_select_sclass0());
    check_select(generator, 1, expected_select_sclass1());
    check_select(generator, 2, expected_select_sclass2());
    check_select(generator, 3, expected_select_mclass0());
    check_select(generator, 4, expected_select_mclass1());
    check_select(generator, 5, expected_select_scalar0());
    check_select(generator, 6, expected_select_scalar1());
    check_select(generator, 7, expected_select_scalar2());
    check_select(generator, 8, expected_select_struct0());
    check_select(generator, 9, expected_select_struct1());
    check_select(generator, 10, expected_select_struct2());
    check_select_stats(generator, make_indices(0, 1, 2), make_indices(3, 4), make_indices(5, 6, 7), make_indices(8, 9, 10));

    const auto expected_flatten = ::expected_flatten();
    const auto expected_columns = make_indices(0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10);

    check_flatten(generator, expected_flatten, expected_columns);

    generator.drop(0);
    check_flatten(generator, drop(expected_flatten, make_indices(0, 1, 2)), expected_columns);

    generator.drop(2);
    check_flatten(generator, drop(expected_flatten, make_indices(0, 1, 2, 5, 6)), expected_columns);

    generator.undrop();
    check_flatten(generator, expected_flatten, expected_columns);

    check_flatten_stats(generator,
        make_indices(5, 5, 5, 10, 10, 5, 5, 10, 10, 10, 5, 5, 5, 5, 10, 5, 4, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(30),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -1, -2, -3, +1, +0, +0, +0, +1, +0, +0, +0, +0, +0, +1, +0, +0),
        make_tensor<scalar_t>(make_dims(30),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            +8, +6, +6, 10, +9, +9, +9, +9, +8, +8, +8, +8, +8, 10, +9, +9),
        make_tensor<scalar_t>(make_dims(30),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.5, 2.0, 1.5, 5.5, 4.5, 4.5, 4.5, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(30),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.027650354097, 3.162277660168, 3.872983346207,
            3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097,
            3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
            3.872983346207, 3.872983346207, 3.872983346207));

    UTEST_CHECK_EQUAL(generator.target(), feature_t{});
    UTEST_CHECK_EQUAL(generator.target_dims(), make_dims(0, 0, 0));

    const auto samples = arange(0, generator.dataset().samples());
    for (const auto exec : {execution::par, execution::seq})
    {
        const auto iterator = flatten_iterator_t{generator, samples, exec, 128};

        UTEST_CHECK_THROW(iterator.loop([&] (tensor_range_t, size_t, tensor4d_cmap_t) {}), std::runtime_error);

        const auto& stats = iterator.targets_stats();
        UTEST_CHECK_EQUAL(std::holds_alternative<scalar_stats_t>(stats), false);
        UTEST_CHECK_EQUAL(std::holds_alternative<sclass_stats_t>(stats), false);
        UTEST_CHECK_EQUAL(std::holds_alternative<mclass_stats_t>(stats), false);
    }
}

UTEST_CASE(sclassification)
{
    const auto dataset = make_dataset(10, 3U);
    const auto generator = make_generator(dataset);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::sclassification);

    UTEST_REQUIRE_EQUAL(generator.features(), 10);
    UTEST_CHECK_EQUAL(generator.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(generator.feature(1), expected_sclass2());
    UTEST_CHECK_EQUAL(generator.feature(2), expected_mclass0());
    UTEST_CHECK_EQUAL(generator.feature(3), expected_mclass1());
    UTEST_CHECK_EQUAL(generator.feature(4), expected_scalar0());
    UTEST_CHECK_EQUAL(generator.feature(5), expected_scalar1());
    UTEST_CHECK_EQUAL(generator.feature(6), expected_scalar2());
    UTEST_CHECK_EQUAL(generator.feature(7), expected_struct0());
    UTEST_CHECK_EQUAL(generator.feature(8), expected_struct1());
    UTEST_CHECK_EQUAL(generator.feature(9), expected_struct2());

    check_select(generator, 0, expected_select_sclass0());
    check_select(generator, 1, expected_select_sclass2());
    check_select(generator, 2, expected_select_mclass0());
    check_select(generator, 3, expected_select_mclass1());
    check_select(generator, 4, expected_select_scalar0());
    check_select(generator, 5, expected_select_scalar1());
    check_select(generator, 6, expected_select_scalar2());
    check_select(generator, 7, expected_select_struct0());
    check_select(generator, 8, expected_select_struct1());
    check_select(generator, 9, expected_select_struct2());
    check_select_stats(generator, make_indices(0, 1), make_indices(2, 3), make_indices(4, 5, 6), make_indices(7, 8, 9));

    const auto expected_flatten = remove(::expected_flatten(), make_indices(3, 4));
    const auto expected_columns = make_indices(0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9);

    check_flatten(generator, expected_flatten, expected_columns);

    check_flatten_stats(generator,
        make_indices(5, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5, 5, 10, 5, 4, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(28),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -1, -2, -3, +1, +0, +0, +0, +1, +0, +0, +0, +0, +0, +1, +0, +0),
        make_tensor<scalar_t>(make_dims(28),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            +8, +6, +6, 10, +9, +9, +9, +9, +8, +8, +8, +8, +8, 10, +9, +9),
        make_tensor<scalar_t>(make_dims(28),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.5, 2.0, 1.5, 5.5, 4.5, 4.5, 4.5, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(28),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.027650354097, 3.162277660168, 3.872983346207,
            3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097,
            3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
            3.872983346207, 3.872983346207, 3.872983346207));

    check_targets(generator, expected_sclass1(), make_dims(2, 1, 1),
        keep(::expected_flatten(), make_indices(3, 4)).reshape(10, 2, 1, 1));
    check_targets_sclass_stats(generator,
        make_indices(5, 5),
        make_tensor<scalar_t>(make_dims(10),
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0));
}

UTEST_CASE(mclassification)
{
    const auto dataset = make_dataset(10, 0U);
    const auto generator = make_generator(dataset);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::mclassification);

    UTEST_REQUIRE_EQUAL(generator.features(), 10);
    UTEST_CHECK_EQUAL(generator.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(generator.feature(1), expected_sclass1());
    UTEST_CHECK_EQUAL(generator.feature(2), expected_sclass2());
    UTEST_CHECK_EQUAL(generator.feature(3), expected_mclass1());
    UTEST_CHECK_EQUAL(generator.feature(4), expected_scalar0());
    UTEST_CHECK_EQUAL(generator.feature(5), expected_scalar1());
    UTEST_CHECK_EQUAL(generator.feature(6), expected_scalar2());
    UTEST_CHECK_EQUAL(generator.feature(7), expected_struct0());
    UTEST_CHECK_EQUAL(generator.feature(8), expected_struct1());
    UTEST_CHECK_EQUAL(generator.feature(9), expected_struct2());

    check_select(generator, 0, expected_select_sclass0());
    check_select(generator, 1, expected_select_sclass1());
    check_select(generator, 2, expected_select_sclass2());
    check_select(generator, 3, expected_select_mclass1());
    check_select(generator, 4, expected_select_scalar0());
    check_select(generator, 5, expected_select_scalar1());
    check_select(generator, 6, expected_select_scalar2());
    check_select(generator, 7, expected_select_struct0());
    check_select(generator, 8, expected_select_struct1());
    check_select(generator, 9, expected_select_struct2());
    check_select_stats(generator, make_indices(0, 1, 2), make_indices(3), make_indices(4, 5, 6), make_indices(7, 8, 9));

    const auto expected_flatten = remove(::expected_flatten(), make_indices(7, 8, 9));
    const auto expected_columns = make_indices(0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9);

    check_flatten(generator, expected_flatten, expected_columns);

    check_flatten_stats(generator,
        make_indices(5, 5, 5, 10, 10, 5, 5, 5, 5, 5, 5, 10, 5, 4, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(27),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -1, -2, -3, +1, +0, +0, +0, +1, +0, +0, +0, +0, +0, +1, +0, +0),
        make_tensor<scalar_t>(make_dims(27),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            +8, +6, +6, 10, +9, +9, +9, +9, +8, +8, +8, +8, +8, 10, +9, +9),
        make_tensor<scalar_t>(make_dims(27),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.5, 2.0, 1.5, 5.5, 4.5, 4.5, 4.5, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(27),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.027650354097, 3.162277660168, 3.872983346207,
            3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097,
            3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
            3.872983346207, 3.872983346207, 3.872983346207));

    check_targets(generator, expected_mclass0(), make_dims(3, 1, 1),
        keep(::expected_flatten(), make_indices(7, 8, 9)).reshape(10, 3, 1, 1));
    check_targets_mclass_stats(generator,
        make_indices(0, 5, 3, 0, 2, 0),
        make_tensor<scalar_t>(make_dims(10),
            1.666666666667, 0.666666666667, 1.111111111111, 0.666666666667, 1.111111111111,
            0.666666666667, 1.666666666667, 0.666666666667, 1.111111111111, 0.666666666667));
}

UTEST_CASE(regression)
{
    const auto dataset = make_dataset(10, 5U);
    const auto generator = make_generator(dataset);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::regression);

    UTEST_REQUIRE_EQUAL(generator.features(), 10);
    UTEST_CHECK_EQUAL(generator.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(generator.feature(1), expected_sclass1());
    UTEST_CHECK_EQUAL(generator.feature(2), expected_sclass2());
    UTEST_CHECK_EQUAL(generator.feature(3), expected_mclass0());
    UTEST_CHECK_EQUAL(generator.feature(4), expected_mclass1());
    UTEST_CHECK_EQUAL(generator.feature(5), expected_scalar1());
    UTEST_CHECK_EQUAL(generator.feature(6), expected_scalar2());
    UTEST_CHECK_EQUAL(generator.feature(7), expected_struct0());
    UTEST_CHECK_EQUAL(generator.feature(8), expected_struct1());
    UTEST_CHECK_EQUAL(generator.feature(9), expected_struct2());

    check_select(generator, 0, expected_select_sclass0());
    check_select(generator, 1, expected_select_sclass1());
    check_select(generator, 2, expected_select_sclass2());
    check_select(generator, 3, expected_select_mclass0());
    check_select(generator, 4, expected_select_mclass1());
    check_select(generator, 5, expected_select_scalar1());
    check_select(generator, 6, expected_select_scalar2());
    check_select(generator, 7, expected_select_struct0());
    check_select(generator, 8, expected_select_struct1());
    check_select(generator, 9, expected_select_struct2());
    check_select_stats(generator, make_indices(0, 1, 2), make_indices(3, 4), make_indices(5, 6), make_indices(7, 8, 9));

    const auto expected_flatten = remove(::expected_flatten(), make_indices(14));
    const auto expected_columns = make_indices(0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9);

    check_flatten(generator, expected_flatten, expected_columns);

    check_flatten_stats(generator,
        make_indices(5, 5, 5, 10, 10, 5, 5, 10, 10, 10, 5, 5, 5, 5, 5, 4, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(29),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -2, -3, +1, +0, +0, +0, +1, +0, +0, +0, +0, +0, +1, +0, +0),
        make_tensor<scalar_t>(make_dims(29),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            +6, +6, 10, +9, +9, +9, +9, +8, +8, +8, +8, +8, 10, +9, +9),
        make_tensor<scalar_t>(make_dims(29),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2.0, 1.5, 5.5, 4.5, 4.5, 4.5, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(29),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.162277660168, 3.872983346207,
            3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097,
            3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
            3.872983346207, 3.872983346207, 3.872983346207));

    check_targets(generator, expected_scalar0(), make_dims(1, 1, 1),
        keep(::expected_flatten(), make_indices(14)).reshape(10, 1, 1, 1));
    check_targets_scalar_stats(generator,  make_indices(10),
        make_tensor<scalar_t>(make_dims(1), -1),
        make_tensor<scalar_t>(make_dims(1), 8),
        make_tensor<scalar_t>(make_dims(1), 3.5),
        make_tensor<scalar_t>(make_dims(1), 3.027650354097));
}

UTEST_CASE(mvregression)
{
    const auto dataset = make_dataset(10, 8U);
    const auto generator = make_generator(dataset);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::regression);

    UTEST_REQUIRE_EQUAL(generator.features(), 10);
    UTEST_CHECK_EQUAL(generator.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(generator.feature(1), expected_sclass1());
    UTEST_CHECK_EQUAL(generator.feature(2), expected_sclass2());
    UTEST_CHECK_EQUAL(generator.feature(3), expected_mclass0());
    UTEST_CHECK_EQUAL(generator.feature(4), expected_mclass1());
    UTEST_CHECK_EQUAL(generator.feature(5), expected_scalar0());
    UTEST_CHECK_EQUAL(generator.feature(6), expected_scalar1());
    UTEST_CHECK_EQUAL(generator.feature(7), expected_scalar2());
    UTEST_CHECK_EQUAL(generator.feature(8), expected_struct1());
    UTEST_CHECK_EQUAL(generator.feature(9), expected_struct2());

    check_select(generator, 0, expected_select_sclass0());
    check_select(generator, 1, expected_select_sclass1());
    check_select(generator, 2, expected_select_sclass2());
    check_select(generator, 3, expected_select_mclass0());
    check_select(generator, 4, expected_select_mclass1());
    check_select(generator, 5, expected_select_scalar0());
    check_select(generator, 6, expected_select_scalar1());
    check_select(generator, 7, expected_select_scalar2());
    check_select(generator, 8, expected_select_struct1());
    check_select(generator, 9, expected_select_struct2());
    check_select_stats(generator, make_indices(0, 1, 2), make_indices(3, 4), make_indices(5, 6, 7), make_indices(8, 9));

    const auto expected_flatten = remove(::expected_flatten(), make_indices(17, 18, 19, 20));
    const auto expected_columns = make_indices(0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9);

    check_flatten(generator, expected_flatten, expected_columns);

    check_flatten_stats(generator,
        make_indices(5, 5, 5, 10, 10, 5, 5, 10, 10, 10, 5, 5, 5, 5, 10, 5, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(26),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            -1, -2, -3, +1, +0, +0, +0, +0, +0, +1, +0, +0),
        make_tensor<scalar_t>(make_dims(26),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            +8, +6, +6, +9, +8, +8, +8, +8, +8, 10, +9, +9),
        make_tensor<scalar_t>(make_dims(26),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.5, 2.0, 1.5, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(26),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            3.027650354097, 3.162277660168, 3.872983346207,
            3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
            3.872983346207, 3.872983346207, 3.872983346207));

    check_targets(generator, expected_struct0(), make_dims(1, 2, 2),
        keep(::expected_flatten(), make_indices(17, 18, 19, 20)).reshape(10, 1, 2, 2));
    check_targets_scalar_stats(generator, make_indices(10, 10, 10, 10),
        make_tensor<scalar_t>(make_dims(4), 1, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(4), 10, 9, 9, 9),
        make_tensor<scalar_t>(make_dims(4), 5.5, 4.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(4), 3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097));
}

UTEST_END_MODULE()