#include <nano/core/sampling.h>
#include <unordered_map>
#include <utest/utest.h>

using namespace nano;

namespace
{
auto make_rngs(const uint64_t seed1 = 42U, const uint64_t seed2 = 42U, const uint64_t seed3 = 43U)
{
    return std::make_tuple(make_rng(seed1), make_rng(seed2), make_rng(seed3));
}

void check_sample_with_replacement(const indices_t& indices, const tensor_size_t expected_count,
                                   const tensor_size_t expected_total)
{
    UTEST_CHECK_EQUAL(indices.size(), expected_count);
    UTEST_CHECK_LESS(indices.max(), expected_total);
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
        const auto count    = 50;
        const auto indices  = nano::sample_with_replacement(samples, count);
        const auto indices1 = nano::sample_with_replacement(samples, count, rng1);
        const auto indices2 = nano::sample_with_replacement(samples, count, rng2);
        const auto indices3 = nano::sample_with_replacement(samples, count, rng3);

        check_sample_with_replacement(indices, count, samples.size());
        check_sample_with_replacement(indices1, count, samples.size());
        check_sample_with_replacement(indices2, count, samples.size());
        check_sample_with_replacement(indices3, count, samples.size());

        UTEST_CHECK_EQUAL(indices1, indices2);
        UTEST_CHECK_NOT_EQUAL(indices1, indices3);
        UTEST_CHECK_NOT_EQUAL(indices, old_indices);
        old_indices = indices;
    }
}

UTEST_CASE(sample_with_replacement_weights)
{
    auto old_indices        = indices_t{};
    auto [rng1, rng2, rng3] = make_rngs();

    const auto weights = make_tensor<scalar_t>(make_dims(5), 3, 1, 1, 2, 3);
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto count    = 2000;
        const auto indices  = nano::sample_with_replacement(weights, count);
        const auto indices1 = nano::sample_with_replacement(weights, count, rng1);
        const auto indices2 = nano::sample_with_replacement(weights, count, rng2);
        const auto indices3 = nano::sample_with_replacement(weights, count, rng3);

        check_sample_with_replacement(indices, count, weights.size());
        check_sample_with_replacement(indices1, count, weights.size());
        check_sample_with_replacement(indices2, count, weights.size());
        check_sample_with_replacement(indices3, count, weights.size());

        for (const auto& values : {indices1, indices2, indices3})
        {
            std::unordered_map<tensor_size_t, tensor_size_t> uvalues;
            for (const auto value : values)
            {
                uvalues[value]++;
            }

            UTEST_REQUIRE_EQUAL(uvalues.size(), static_cast<size_t>(weights.size()));
            UTEST_CHECK_GREATER(uvalues[0], 540);
            UTEST_CHECK_GREATER(uvalues[1], 150);
            UTEST_CHECK_GREATER(uvalues[2], 150);
            UTEST_CHECK_GREATER(uvalues[3], 340);
            UTEST_CHECK_GREATER(uvalues[4], 540);
        }

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
