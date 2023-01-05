#include <nano/model/util.h>
#include <utest/utest.h>

using namespace nano;

static void check_sample_with_replacement(const indices_t& indices)
{
    UTEST_CHECK_EQUAL(indices.size(), 50);
    UTEST_CHECK_LESS(indices.max(), 120);
    UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
    UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
}

static void check_sample_without_replacement(const indices_t& indices)
{
    UTEST_CHECK_EQUAL(indices.size(), 60);
    UTEST_CHECK_LESS(indices.max(), 140);
    UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
    UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
    UTEST_CHECK(std::adjacent_find(begin(indices), end(indices)) == end(indices));
}

UTEST_BEGIN_MODULE(test_model_util)

UTEST_CASE(sample_with_replacement)
{
    auto old_indices = indices_t{};
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto indices = nano::sample_with_replacement(120, 50);

        check_sample_with_replacement(indices);
        UTEST_CHECK_NOT_EQUAL(indices, old_indices);
        old_indices = indices;
    }
}

UTEST_CASE(sample_with_replacement_samples)
{
    const auto samples = arange(0, 120);
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto indices = nano::sample_with_replacement(samples, 50);
        check_sample_with_replacement(indices);
    }
}

UTEST_CASE(sample_with_replacement_fixed_seed)
{
    for (auto trial = 0; trial < 10; ++trial)
    {
        const auto indices1 = nano::sample_with_replacement(120, 50, static_cast<uint64_t>(trial * 31 + 1));
        const auto indices2 = nano::sample_with_replacement(120, 50, static_cast<uint64_t>(trial * 31 + 1));
        const auto indices3 = nano::sample_with_replacement(120, 50, static_cast<uint64_t>(trial * 31 + 2));

        check_sample_with_replacement(indices1);
        check_sample_with_replacement(indices2);
        check_sample_with_replacement(indices3);
        UTEST_CHECK_EQUAL(indices1, indices2);
        UTEST_CHECK_NOT_EQUAL(indices1, indices3);
    }
}

UTEST_CASE(sample_without_replacement)
{
    auto old_indices = indices_t{};
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto indices = nano::sample_without_replacement(140, 60);

        check_sample_without_replacement(indices);
        UTEST_CHECK_NOT_EQUAL(indices, old_indices);
        old_indices = indices;
    }
}

UTEST_CASE(sample_without_replacement_all)
{
    const auto indices = nano::sample_without_replacement(100, 100);

    UTEST_CHECK_EQUAL(indices, arange(0, 100));
}

UTEST_CASE(sample_without_replacement_samples)
{
    const auto samples = arange(0, 140);
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto indices = nano::sample_without_replacement(samples, 60);
        check_sample_without_replacement(indices);
    }
}

UTEST_CASE(sample_without_replacement_fixed_seed)
{
    for (auto trial = 0; trial < 10; ++trial)
    {
        const auto indices1 = nano::sample_without_replacement(140, 60, static_cast<uint64_t>(trial * 31 + 1));
        const auto indices2 = nano::sample_without_replacement(140, 60, static_cast<uint64_t>(trial * 31 + 1));
        const auto indices3 = nano::sample_without_replacement(140, 60, static_cast<uint64_t>(trial * 31 + 2));

        check_sample_without_replacement(indices1);
        check_sample_without_replacement(indices2);
        check_sample_without_replacement(indices3);
        UTEST_CHECK_EQUAL(indices1, indices2);
        UTEST_CHECK_NOT_EQUAL(indices1, indices3);
    }
}

UTEST_END_MODULE()
