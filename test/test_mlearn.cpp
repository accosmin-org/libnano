#include <utest/utest.h>
#include <nano/mlearn.h>

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

UTEST_CASE(split_valid)
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

UTEST_END_MODULE()
