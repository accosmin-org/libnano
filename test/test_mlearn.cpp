#include <utest/utest.h>
#include <nano/mlearn/split.h>
#include <nano/mlearn/elemwise.h>

using namespace nano;

template <typename tscalar>
std::ostream& operator<<(std::ostream& os, const std::vector<tscalar>& values)
{
    os << "{";
    for (const auto& value : values)
    {
        os << "{" << value << "}";
    }
    return os << "}";
}

UTEST_BEGIN_MODULE(test_mlearn)

UTEST_CASE(split2)
{
    const auto count = 120;
    const auto percentage1 = 60;
    const auto percentage2 = 100 - percentage1;

    indices_t set1, set2;
    std::tie(set1, set2) = nano::split2(count, percentage1);

    UTEST_CHECK_EQUAL(set1.size(), percentage1 * count / 100);
    UTEST_CHECK_EQUAL(set2.size(), percentage2 * count / 100);

    UTEST_CHECK_GREATER_EQUAL(set1.minCoeff(), 0);
    UTEST_CHECK_GREATER_EQUAL(set2.minCoeff(), 0);

    UTEST_CHECK_LESS(set1.maxCoeff(), count);
    UTEST_CHECK_LESS(set2.maxCoeff(), count);

    UTEST_CHECK(std::is_sorted(begin(set1), end(set1)));
    UTEST_CHECK(std::is_sorted(begin(set2), end(set2)));

    UTEST_CHECK(std::adjacent_find(begin(set1), end(set1)) == end(set1));
    UTEST_CHECK(std::adjacent_find(begin(set2), end(set2)) == end(set2));
}

UTEST_CASE(split3)
{
    const auto count = 120;
    const auto percentage1 = 60;
    const auto percentage2 = 30;
    const auto percentage3 = 100 - percentage1 - percentage2;

    indices_t set1, set2, set3;
    std::tie(set1, set2, set3) = nano::split3(count, percentage1, percentage2);

    UTEST_CHECK_EQUAL(set1.size(), percentage1 * count / 100);
    UTEST_CHECK_EQUAL(set2.size(), percentage2 * count / 100);
    UTEST_CHECK_EQUAL(set3.size(), percentage3 * count / 100);

    UTEST_CHECK_GREATER_EQUAL(set1.minCoeff(), 0);
    UTEST_CHECK_GREATER_EQUAL(set2.minCoeff(), 0);
    UTEST_CHECK_GREATER_EQUAL(set3.minCoeff(), 0);

    UTEST_CHECK_LESS(set1.maxCoeff(), count);
    UTEST_CHECK_LESS(set2.maxCoeff(), count);
    UTEST_CHECK_LESS(set3.maxCoeff(), count);

    UTEST_CHECK(std::is_sorted(begin(set1), end(set1)));
    UTEST_CHECK(std::is_sorted(begin(set2), end(set2)));
    UTEST_CHECK(std::is_sorted(begin(set3), end(set3)));

    UTEST_CHECK(std::adjacent_find(begin(set1), end(set1)) == end(set1));
    UTEST_CHECK(std::adjacent_find(begin(set2), end(set2)) == end(set2));
    UTEST_CHECK(std::adjacent_find(begin(set3), end(set3)) == end(set3));
}

UTEST_CASE(sample_with_replacement)
{
    for (auto trial = 0; trial < 100; ++ trial)
    {
        const auto indices = nano::sample_with_replacement(120, 50);

        UTEST_CHECK_EQUAL(indices.size(), 60);
        UTEST_CHECK_LESS(indices.maxCoeff(), 120);
        UTEST_CHECK_GREATER_EQUAL(indices.minCoeff(), 0);
        UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
    }
}

UTEST_CASE(sample_without_replacement)
{
    for (auto trial = 0; trial < 100; ++ trial)
    {
        const auto indices = nano::sample_without_replacement(140, 50);

        UTEST_CHECK_EQUAL(indices.size(), 70);
        UTEST_CHECK_LESS(indices.maxCoeff(), 140);
        UTEST_CHECK_GREATER_EQUAL(indices.minCoeff(), 0);
        UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
        UTEST_CHECK(std::adjacent_find(begin(indices), end(indices)) == end(indices));
    }
}

UTEST_CASE(sample_without_replacement_all)
{
    const auto indices = nano::sample_without_replacement(100, 100);

    UTEST_CHECK_EQUAL(indices, indices_t::LinSpaced(100, 0, 100));
}

UTEST_CASE(split_valid2)
{
    const auto split = split_t{nano::split2(80, 60), indices_t::LinSpaced(20, 80, 100)};

    UTEST_CHECK(split.valid(100));
    UTEST_CHECK_EQUAL(split.indices(protocol::train).size(), 48);
    UTEST_CHECK_EQUAL(split.indices(protocol::valid).size(), 32);
    UTEST_CHECK_EQUAL(split.indices(protocol::test).size(), 20);

    UTEST_CHECK(!split.valid(90));
}

UTEST_CASE(split_valid3)
{
    const auto split = split_t{nano::split3(100, 60, 30)};

    UTEST_CHECK(split.valid(100));
    UTEST_CHECK_EQUAL(split.indices(protocol::train).size(), 60);
    UTEST_CHECK_EQUAL(split.indices(protocol::valid).size(), 30);
    UTEST_CHECK_EQUAL(split.indices(protocol::test).size(), 10);

    UTEST_CHECK(!split.valid(90));
}

UTEST_CASE(split_invalid_empty)
{
    const auto split = split_t{};

    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_tr_out_of_range)
{
    auto split = split_t{nano::split3(100, 60, 30)};

    split.indices(protocol::train)(0) = -1;
    UTEST_CHECK(!split.valid(100));

    split.indices(protocol::train)(0) = 101;
    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_vd_out_of_range)
{
    auto split = split_t{nano::split3(100, 60, 30)};

    split.indices(protocol::valid)(0) = -1;
    UTEST_CHECK(!split.valid(100));

    split.indices(protocol::valid)(0) = 101;
    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_te_out_of_range)
{
    auto split = split_t{nano::split3(100, 60, 30)};

    split.indices(protocol::test)(0) = -1;
    UTEST_CHECK(!split.valid(100));

    split.indices(protocol::test)(0) = 101;
    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_tr_vd_intersects)
{
    auto split = split_t{nano::split3(100, 60, 30)};
    split.indices(protocol::valid) = split.indices(protocol::train).segment(0, split.indices(protocol::valid).size());

    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(split_invalid_tr_te_intersects)
{
    auto split = split_t{nano::split3(100, 60, 30)};
    split.indices(protocol::test) = split.indices(protocol::train).segment(0, split.indices(protocol::test).size());

    UTEST_CHECK(!split.valid(100));
}

UTEST_CASE(scale)
{
    const std::array<scalar_t, 3> min = {{-1, +1, +2}};
    const std::array<scalar_t, 3> max = {{+1, +3, +7}};
    const std::array<scalar_t, 3> mean = {{0.0, 1.1, 5.1}};
    const std::array<scalar_t, 3> stdev = {{0.1, 0.2, 0.8}};

    elemwise_stats_t stats;
    stats.set(
        map_tensor(min.data(), 3, 1, 1),
        map_tensor(max.data(), 3, 1, 1),
        map_tensor(mean.data(), 3, 1, 1),
        map_tensor(stdev.data(), 3, 1, 1));

    const std::array<scalar_t, 15> original =
    {{
        -1.0, +1.0, +2.0,
        -0.5, +1.5, +3.0,
        -0.0, +2.0, +4.0,
        +0.5, +2.5, +5.0,
        +1.0, +3.0, +6.0
    }};

    const std::array<scalar_t, 15> normed_none =
    {{
        -1.0, +1.0, +2.0,
        -0.5, +1.5, +3.0,
        -0.0, +2.0, +4.0,
        +0.5, +2.5, +5.0,
        +1.0, +3.0, +6.0
    }};

    const std::array<scalar_t, 15> normed_mean =
    {{
        -0.50, -0.05, -3.1/5.0,
        -0.25, +0.20, -2.1/5.0,
        -0.00, +0.45, -1.1/5.0,
        +0.25, +0.70, -0.1/5.0,
        +0.50, +0.95, +0.9/5.0
    }};

    const std::array<scalar_t, 15> normed_minmax =
    {{
        +0.00, +0.00, +0.00,
        +0.25, +0.25, +0.20,
        +0.50, +0.50, +0.40,
        +0.75, +0.75, +0.60,
        +1.00, +1.00, +0.80
    }};

    const std::array<scalar_t, 15> normed_standard =
    {{
        -10.0, -0.50, -3.1/0.8,
         -5.0, +2.00, -2.1/0.8,
         +0.0, +4.50, -1.1/0.8,
         +5.0, +7.00, -0.1/0.8,
        +10.0, +9.50, +0.9/0.8
    }};

    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(normalization::none, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_none.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(normalization::mean, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_mean.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(normalization::minmax, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_minmax.data(), 15), epsilon1<scalar_t>());
    }
    {
        auto inputs = tensor4d_t{map_tensor(original.data(), 5, 3, 1, 1)};
        UTEST_CHECK_NOTHROW(stats.scale(normalization::standard, inputs));
        UTEST_CHECK_EIGEN_CLOSE(inputs.vector(), map_vector(normed_standard.data(), 15), epsilon1<scalar_t>());
    }
}

UTEST_END_MODULE()
