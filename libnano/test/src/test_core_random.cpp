#include <utest/utest.h>
#include "core/random.h"

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

UTEST_BEGIN_MODULE(test_core_random)

UTEST_CASE(split2)
{
        const auto count = size_t(120);
        const auto value1 = 7;
        const auto value2 = 5;

        const auto percentage_value1 = size_t(60);
        const auto percentage_value2 = size_t(100) - percentage_value1;

        const auto values = nano::split2(count, value1, percentage_value1, value2);

        UTEST_REQUIRE_EQUAL(values.size(), count);
        UTEST_CHECK_EQUAL(std::count(values.begin(), values.end(), value1), percentage_value1 * count / 100);
        UTEST_CHECK_EQUAL(std::count(values.begin(), values.end(), value2), percentage_value2 * count / 100);
}

UTEST_CASE(split3)
{
        const auto count = size_t(420);
        const auto value1 = 7;
        const auto value2 = 5;
        const auto value3 = 3;

        const auto percentage_value1 = size_t(60);
        const auto percentage_value2 = size_t(30);
        const auto percentage_value3 = size_t(100) - percentage_value1 - percentage_value2;

        const auto values = nano::split3(count, value1, percentage_value1, value2, percentage_value2, value3);

        UTEST_REQUIRE_EQUAL(values.size(), count);
        UTEST_CHECK_EQUAL(std::count(values.begin(), values.end(), value1), percentage_value1 * count / 100);
        UTEST_CHECK_EQUAL(std::count(values.begin(), values.end(), value2), percentage_value2 * count / 100);
        UTEST_CHECK_EQUAL(std::count(values.begin(), values.end(), value3), percentage_value3 * count / 100);
}

UTEST_CASE(sample_with_replacement)
{
        for (auto trial = 0; trial < 100; ++ trial)
        {
                const auto indices = nano::sample_with_replacement(100, 50);
                UTEST_REQUIRE_EQUAL(indices.size(), 50);
                UTEST_CHECK(std::is_sorted(indices.begin(), indices.end()));
                UTEST_CHECK_LESS(*std::max_element(indices.begin(), indices.end()), 100);
                UTEST_CHECK_GREATER_EQUAL(*std::min_element(indices.begin(), indices.end()), 0);
        }
}

UTEST_CASE(sample_without_replacement)
{
        for (auto trial = 0; trial < 100; ++ trial)
        {
                auto indices = nano::sample_without_replacement(100, 50);
                UTEST_REQUIRE_EQUAL(indices.size(), 50);
                UTEST_CHECK(std::is_sorted(indices.begin(), indices.end()));
                UTEST_CHECK_LESS(*std::max_element(indices.begin(), indices.end()), 100);
                UTEST_CHECK_GREATER_EQUAL(*std::min_element(indices.begin(), indices.end()), 0);
                UTEST_CHECK(std::unique(indices.begin(), indices.end()) == indices.end());
        }
}

UTEST_CASE(sample_without_replacement_all)
{
        const auto indices = nano::sample_without_replacement(100, 100);
        std::vector<int> all_indices(100);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        UTEST_CHECK_EQUAL(indices, all_indices);
}

UTEST_END_MODULE()
