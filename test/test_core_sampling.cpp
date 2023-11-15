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
    UTEST_CHECK(std::is_sorted(std::begin(indices), std::end(indices)));
}

void check_sample_without_replacement(const indices_t& indices)
{
    UTEST_CHECK_EQUAL(indices.size(), 60);
    UTEST_CHECK_LESS(indices.max(), 140);
    UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
    UTEST_CHECK(std::is_sorted(std::begin(indices), std::end(indices)));
    UTEST_CHECK(std::adjacent_find(std::begin(indices), std::end(indices)) == std::end(indices));
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

    const auto samples = arange(0, 5);
    const auto weights = make_tensor<scalar_t>(make_dims(5), 3, 1, 1, 2, 3);
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto count    = 2000;
        const auto indices  = nano::sample_with_replacement(samples, weights, count);
        const auto indices1 = nano::sample_with_replacement(samples, weights, count, rng1);
        const auto indices2 = nano::sample_with_replacement(samples, weights, count, rng2);
        const auto indices3 = nano::sample_with_replacement(samples, weights, count, rng3);

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

UTEST_CASE(sample_from_ball)
{
    for (const tensor_size_t dims : {2, 3, 11})
    {
        auto       rng    = make_rng();
        const auto x0     = make_random_vector<scalar_t>(dims, -1.0, +1.0);
        const auto radius = urand(0.1, 1.0, rng);

        // check all points are inside the ball
        for (tensor_size_t trial = 0; trial < dims + 1; ++trial)
        {
            const auto x = nano::sample_from_ball(x0, radius, rng);
            UTEST_REQUIRE_EQUAL(x.size(), x0.size());
            UTEST_CHECK_LESS_EQUAL((x - x0).lpNorm<2>(), radius);
        }
    }
}

UTEST_CASE(sample_from_ball2D)
{
    const auto dims = tensor_size_t{2};
    const auto size = tensor_size_t{1000};

    auto       rng    = make_rng();
    const auto x0     = make_random_vector<scalar_t>(dims, size / 4, size * 3 / 4);
    const auto radius = urand(size / 8, size / 5, rng);

    auto n_hits = tensor_size_t{0};
    for (tensor_size_t r = 0; r < size; ++r)
    {
        for (tensor_size_t c = 0; c < size; ++c)
        {
            const auto r0 = static_cast<tensor_size_t>(std::round(x0(0)));
            const auto c0 = static_cast<tensor_size_t>(std::round(x0(1)));
            if ((r - r0) * (r - r0) + (c - c0) * (c - c0) <= radius)
            {
                ++n_hits;
            }
        }
    }

    // check that all points are inside the ball and approximatly uniformly distributed
    auto hits = std::unordered_map<tensor_size_t, tensor_size_t>{};
    for (tensor_size_t trial = 0; trial < n_hits; ++trial)
    {
        const auto x = nano::sample_from_ball(x0, static_cast<scalar_t>(radius), rng);
        UTEST_REQUIRE_EQUAL(x.size(), x0.size());
        UTEST_CHECK_LESS_EQUAL((x - x0).lpNorm<2>(), static_cast<scalar_t>(radius));

        const auto r = static_cast<tensor_size_t>(std::round(x(0)));
        const auto c = static_cast<tensor_size_t>(std::round(x(1)));
        hits[r * size + c] += 1;
    }
    UTEST_CHECK_GREATER(static_cast<tensor_size_t>(hits.size()), 9 * n_hits / 10);
}

UTEST_END_MODULE()
