#include <nano/core/sampling.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
auto make_rngs(const uint64_t seed1 = 42U, const uint64_t seed2 = 42U, const uint64_t seed3 = 43U)
{
    return std::make_tuple(make_rng(seed1), make_rng(seed2), make_rng(seed3));
}

void check_sample_with_replacement(const indices_t& indices)
{
    UTEST_CHECK_EQUAL(indices.size(), 50);
    UTEST_CHECK_LESS(indices.max(), 120);
    UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
    UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
}

void check_sample_without_replacement(const indices_t& indices)
{
    UTEST_CHECK_EQUAL(indices.size(), 60);
    UTEST_CHECK_LESS(indices.max(), 140);
    UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
    UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
    UTEST_CHECK(std::adjacent_find(begin(indices), end(indices)) == end(indices));
}
} // namespace

UTEST_BEGIN_MODULE(test_core_sampling)

UTEST_CASE(sample_with_replacement)
{
    auto old_indices        = indices_t{};
    auto [rng1, rng2, rng3] = make_rngs();

    const auto samples = arange(0, 120);
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto indices  = nano::sample_with_replacement(samples, 50);
        const auto indices1 = nano::sample_with_replacement(samples, 50, rng1);
        const auto indices2 = nano::sample_with_replacement(samples, 50, rng2);
        const auto indices3 = nano::sample_with_replacement(samples, 50, rng3);

        check_sample_with_replacement(indices);
        check_sample_with_replacement(indices1);
        check_sample_with_replacement(indices2);
        check_sample_with_replacement(indices3);

        UTEST_CHECK_EQUAL(indices1, indices2);
        UTEST_CHECK_NOT_EQUAL(indices1, indices3);
        UTEST_CHECK_NOT_EQUAL(indices, old_indices);
        old_indices = indices;
    }
}

UTEST_CASE(sample_without_replacement)
{
    auto old_indices        = indices_t{};
    auto [rng1, rng2, rng3] = make_rngs();

    const auto samples = arange(0, 140);
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto indices  = nano::sample_without_replacement(samples, 60);
        const auto indices1 = nano::sample_without_replacement(samples, 60, rng1);
        const auto indices2 = nano::sample_without_replacement(samples, 60, rng2);
        const auto indices3 = nano::sample_without_replacement(samples, 60, rng3);

        check_sample_without_replacement(indices);
        check_sample_without_replacement(indices1);
        check_sample_without_replacement(indices2);
        check_sample_without_replacement(indices3);

        UTEST_CHECK_EQUAL(indices1, indices2);
        UTEST_CHECK_NOT_EQUAL(indices1, indices3);
        UTEST_CHECK_NOT_EQUAL(indices, old_indices);
        old_indices = indices;
    }
}

UTEST_CASE(sample_without_replacement_all)
{
    auto [rng1, rng2, rng3] = make_rngs();

    const auto samples  = arange(0, 100);
    const auto indices  = nano::sample_without_replacement(samples, 100);
    const auto indices1 = nano::sample_without_replacement(samples, 100, rng1);
    const auto indices2 = nano::sample_without_replacement(samples, 100, rng2);
    const auto indices3 = nano::sample_without_replacement(samples, 100, rng3);

    UTEST_CHECK_EQUAL(indices, samples);
    UTEST_CHECK_EQUAL(indices1, samples);
    UTEST_CHECK_EQUAL(indices2, samples);
    UTEST_CHECK_EQUAL(indices3, samples);
}

UTEST_END_MODULE()
