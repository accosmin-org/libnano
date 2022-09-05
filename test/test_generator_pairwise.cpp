#include "fixture/generator.h"
#include "fixture/generator_dataset.h"
#include <nano/generator/pairwise.h>
#include <utest/utest.h>

using namespace nano;

class scalar_scalar_to_scalar_t : public pairwise_input_scalar_scalar_t, public generated_scalar_t
{
public:
    scalar_scalar_to_scalar_t()
        : pairwise_input_scalar_scalar_t("gg")
    {
    }

    explicit scalar_scalar_to_scalar_t(indices_t features)
        : pairwise_input_scalar_scalar_t("gg", std::move(features))
    {
    }

    scalar_scalar_to_scalar_t(indices_t features1, indices_t features2)
        : pairwise_input_scalar_scalar_t("gg", std::move(features1), std::move(features2))
    {
    }

    feature_t feature(tensor_size_t ifeature) const override { return make_scalar_feature(ifeature, "sum"); }

    static auto process(tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& values1, const auto& values2)
        { return static_cast<scalar_t>(values1(0)) + static_cast<scalar_t>(values2(0)); };
        return std::make_tuple(process, colsize);
    }
};

class scalar_scalar_to_struct_t : public pairwise_input_scalar_scalar_t, public generated_struct_t
{
public:
    scalar_scalar_to_struct_t()
        : pairwise_input_scalar_scalar_t("gg")
    {
    }

    explicit scalar_scalar_to_struct_t(indices_t features)
        : pairwise_input_scalar_scalar_t("gg", std::move(features))
    {
    }

    scalar_scalar_to_struct_t(indices_t features1, indices_t features2)
        : pairwise_input_scalar_scalar_t("gg", std::move(features1), std::move(features2))
    {
    }

    feature_t feature(tensor_size_t ifeature) const override
    {
        return make_struct_feature(ifeature, "pow", make_dims(3, 1, 1));
    }

    static auto process(tensor_size_t)
    {
        const auto colsize = tensor_size_t{3};
        const auto process = [=](const auto& values1, const auto& values2, auto&& structured)
        {
            const auto value1 = static_cast<scalar_t>(values1(0));
            const auto value2 = static_cast<scalar_t>(values2(0));
            structured(0)     = value1 * value1;
            structured(1)     = value1 * value2;
            structured(2)     = value2 * value2;
        };
        return std::make_tuple(process, colsize);
    }
};

class scalar_scalar_to_sclass_t : public pairwise_input_scalar_scalar_t, public generated_sclass_t
{
public:
    scalar_scalar_to_sclass_t()
        : pairwise_input_scalar_scalar_t("gg")
    {
    }

    explicit scalar_scalar_to_sclass_t(indices_t features)
        : pairwise_input_scalar_scalar_t("gg", std::move(features))
    {
    }

    scalar_scalar_to_sclass_t(indices_t features1, indices_t features2)
        : pairwise_input_scalar_scalar_t("gg", std::move(features1), std::move(features2))
    {
    }

    feature_t feature(tensor_size_t ifeature) const override
    {
        return make_sclass_feature(ifeature, "sign", strings_t{"neg", "pos"});
    }

    static auto process(tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& values1, const auto& values2)
        {
            const auto value1 = static_cast<scalar_t>(values1(0));
            const auto value2 = static_cast<scalar_t>(values2(0));
            return (value1 < 0.0 || value2 < 0.0) ? 0 : 1;
        };
        return std::make_tuple(process, colsize);
    }
};

class scalar_scalar_to_mclass_t : public pairwise_input_scalar_scalar_t, public generated_mclass_t
{
public:
    scalar_scalar_to_mclass_t()
        : pairwise_input_scalar_scalar_t("gg")
    {
    }

    explicit scalar_scalar_to_mclass_t(indices_t features)
        : pairwise_input_scalar_scalar_t("gg", std::move(features))
    {
    }

    scalar_scalar_to_mclass_t(indices_t features1, indices_t features2)
        : pairwise_input_scalar_scalar_t("gg", std::move(features1), std::move(features2))
    {
    }

    feature_t feature(tensor_size_t ifeature) const override
    {
        return make_mclass_feature(ifeature, "mod", strings_t{"mod2", "mod3"});
    }

    static auto process(tensor_size_t)
    {
        const auto colsize = tensor_size_t{2};
        const auto process = [=](const auto& values1, const auto& values2, auto&& mclass)
        {
            const auto value1 = static_cast<int>(static_cast<scalar_t>(values1(0)));
            const auto value2 = static_cast<int>(static_cast<scalar_t>(values2(0)));
            mclass(0)         = (value1 + value2) % 2 == 0 ? 0x01 : 0x00;
            mclass(1)         = (value1 + value2) % 3 == 0 ? 0x01 : 0x00;
        };
        return std::make_tuple(process, colsize);
    }
};

class sclass_sclass_to_scalar_t : public pairwise_input_sclass_sclass_t, public generated_scalar_t
{
public:
    sclass_sclass_to_scalar_t()
        : pairwise_input_sclass_sclass_t("gg")
    {
    }

    explicit sclass_sclass_to_scalar_t(indices_t features)
        : pairwise_input_sclass_sclass_t("gg", std::move(features))
    {
    }

    sclass_sclass_to_scalar_t(indices_t features1, indices_t features2)
        : pairwise_input_sclass_sclass_t("gg", std::move(features1), std::move(features2))
    {
    }

    feature_t feature(tensor_size_t ifeature) const override { return make_scalar_feature(ifeature, "sum"); }

    static auto process(tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& value1, const auto& value2)
        { return static_cast<scalar_t>(value1) + static_cast<scalar_t>(value2); };
        return std::make_tuple(process, colsize);
    }
};

UTEST_BEGIN_MODULE(test_generator_pairwise)

UTEST_CASE(scalar_scalar)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    add_generator<pairwise_generator_t<scalar_scalar_to_scalar_t>>(generator);
    add_generator<pairwise_generator_t<scalar_scalar_to_struct_t>>(generator, make_indices(6));
    add_generator<pairwise_generator_t<scalar_scalar_to_sclass_t>>(generator);
    add_generator<pairwise_generator_t<scalar_scalar_to_mclass_t>>(generator, make_indices(6), make_indices(6, 7));

    UTEST_REQUIRE_EQUAL(generator.features(), 15);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sum(scalar0,scalar0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sum(scalar0,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"sum(scalar0,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"sum(scalar1,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"sum(scalar1,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"sum(scalar2,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(6),
                      feature_t{"pow(scalar1,scalar1)"}.scalar(feature_type::float64, make_dims(3, 1, 1)));
    UTEST_CHECK_EQUAL(generator.feature(7), feature_t{"sign(scalar0,scalar0)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(8), feature_t{"sign(scalar0,scalar1)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(9), feature_t{"sign(scalar0,scalar2)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(10), feature_t{"sign(scalar1,scalar1)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(11), feature_t{"sign(scalar1,scalar2)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(12), feature_t{"sign(scalar2,scalar2)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(generator.feature(13), feature_t{"mod(scalar1,scalar1)"}.mclass(strings_t{"mod2", "mod3"}));
    UTEST_CHECK_EQUAL(generator.feature(14), feature_t{"mod(scalar1,scalar2)"}.mclass(strings_t{"mod2", "mod3"}));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(10), -2, 0, 2, 4, 6, 8, 10, 12, 14, 16));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(10), -3, N, 1, N, 5, N, +9, Na, 13, Na));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(10), -4, N, N, 2, N, N, +8, Na, Na, 14));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(10), -4, N, 0, N, 4, N, +8, Na, 12, Na));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(10), -5, N, N, N, N, N, +7, Na, Na, Na));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(10), -6, N, N, 0, N, N, +6, Na, Na, 12));
    check_select(generator, 6,
                 make_tensor<scalar_t>(make_dims(10, 3, 1, 1), 4, 4, 4, N, N, N, 0, 0, 0, N, N, N, 4, 4, 4, N, N, N, 16,
                                       16, 16, N, N, N, 36, 36, 36, N, N, N));
    check_select(generator, 7, make_tensor<int32_t>(make_dims(10), +0, +1, +1, +1, +1, +1, +1, +1, +1, +1));
    check_select(generator, 8, make_tensor<int32_t>(make_dims(10), +0, -1, +1, -1, +1, -1, +1, -1, +1, -1));
    check_select(generator, 9, make_tensor<int32_t>(make_dims(10), +0, -1, -1, +1, -1, -1, +1, -1, -1, +1));
    check_select(generator, 10, make_tensor<int32_t>(make_dims(10), +0, -1, +1, -1, +1, -1, +1, -1, +1, -1));
    check_select(generator, 11, make_tensor<int32_t>(make_dims(10), +0, -1, -1, -1, -1, -1, +1, -1, -1, -1));
    check_select(generator, 12, make_tensor<int32_t>(make_dims(10), +0, -1, -1, +1, -1, -1, +1, -1, -1, +1));
    check_select(generator, 13,
                 make_tensor<int8_t>(make_dims(10, 2), +1, +0, -1, -1, +1, +1, -1, -1, +1, +0, -1, -1, +1, +0, -1, -1,
                                     +1, +1, -1, -1));
    check_select(generator, 14,
                 make_tensor<int8_t>(make_dims(10, 2), +0, +0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +0, +0, -1, -1,
                                     -1, -1, -1, -1));

    generator.drop(0);
    check_flatten(generator,
                  make_tensor<scalar_t>(make_dims(10, 19), Na, -3, -4, -4, -5, -6, +4, +4, +4, +1, +1, +1, +1, +1, +1,
                                        +1, -1, -1, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na,
                                        Na, Na, Na, Na, +1, Na, +0, Na, Na, +0, +0, +0, -1, -1, Na, -1, Na, Na, +1, +1,
                                        Na, Na, Na, Na, +2, Na, Na, +0, Na, Na, Na, -1, Na, -1, Na, Na, -1, Na, Na, Na,
                                        Na, Na, +5, Na, +4, Na, Na, +4, +4, +4, -1, -1, Na, -1, Na, Na, +1, -1, Na, Na,
                                        Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, Na,
                                        +9, +8, +8, +7, +6, 16, 16, 16, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, Na, Na,
                                        Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, Na, 13, Na,
                                        12, Na, Na, 36, 36, 36, -1, -1, Na, -1, Na, Na, +1, +1, Na, Na, Na, Na, 14, Na,
                                        Na, 12, Na, Na, Na, -1, Na, -1, Na, Na, -1, Na, Na, Na, Na),
                  make_indices(0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14));

    generator.drop(6);
    check_flatten(generator,
                  make_tensor<scalar_t>(make_dims(10, 19), Na, -3, -4, -4, -5, -6, Na, Na, Na, +1, +1, +1, +1, +1, +1,
                                        +1, -1, -1, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na,
                                        Na, Na, Na, Na, +1, Na, +0, Na, Na, Na, Na, Na, -1, -1, Na, -1, Na, Na, +1, +1,
                                        Na, Na, Na, Na, +2, Na, Na, +0, Na, Na, Na, -1, Na, -1, Na, Na, -1, Na, Na, Na,
                                        Na, Na, +5, Na, +4, Na, Na, Na, Na, Na, -1, -1, Na, -1, Na, Na, +1, -1, Na, Na,
                                        Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, Na,
                                        +9, +8, +8, +7, +6, Na, Na, Na, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, Na, Na,
                                        Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, Na, 13, Na,
                                        12, Na, Na, Na, Na, Na, -1, -1, Na, -1, Na, Na, +1, +1, Na, Na, Na, Na, 14, Na,
                                        Na, 12, Na, Na, Na, -1, Na, -1, Na, Na, -1, Na, Na, Na, Na),
                  make_indices(0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14));
}

UTEST_CASE(sclass_sclass)
{
    const auto dataset = make_dataset(10, string_t::npos);

    auto generator = dataset_generator_t{dataset};
    add_generator<pairwise_generator_t<sclass_sclass_to_scalar_t>>(generator);

    UTEST_REQUIRE_EQUAL(generator.features(), 6);
    UTEST_CHECK_EQUAL(generator.feature(0), feature_t{"sum(sclass0,sclass0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(1), feature_t{"sum(sclass0,sclass1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(2), feature_t{"sum(sclass0,sclass2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(3), feature_t{"sum(sclass1,sclass1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(4), feature_t{"sum(sclass1,sclass2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(generator.feature(5), feature_t{"sum(sclass2,sclass2)"}.scalar(feature_type::float64));

    check_select(generator, 0, make_tensor<scalar_t>(make_dims(10), 4, N, 2, N, 0, N, 4, N, 2, N));
    check_select(generator, 1, make_tensor<scalar_t>(make_dims(10), 3, N, 2, N, 1, N, 3, N, 2, N));
    check_select(generator, 2, make_tensor<scalar_t>(make_dims(10), 2, N, 1, N, 0, N, 2, N, 1, N));
    check_select(generator, 3, make_tensor<scalar_t>(make_dims(10), 2, 0, 2, 0, 2, 0, 2, 0, 2, 0));
    check_select(generator, 4, make_tensor<scalar_t>(make_dims(10), 1, N, 1, N, 1, N, 1, N, 1, N));
    check_select(generator, 5, make_tensor<scalar_t>(make_dims(10), 0, N, 0, N, 0, N, 0, N, 0, N));

    check_flatten(generator,
                  make_tensor<scalar_t>(make_dims(10, 6), 4, 3, 2, 2, 1, 0, N, N, N, 0, N, N, 2, 2, 1, 2, 1, 0, N, N, N,
                                        0, N, N, 0, 1, 0, 2, 1, 0, N, N, N, 0, N, N, 4, 3, 2, 2, 1, 0, N, N, N, 0, N, N,
                                        2, 2, 1, 2, 1, 0, N, N, N, 0, N, N),
                  make_indices(0, 1, 2, 3, 4, 5));
}

UTEST_END_MODULE()
