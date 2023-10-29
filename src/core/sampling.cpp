#include <nano/core/sampling.h>

using namespace nano;

indices_t nano::sample_with_replacement(sample_indices_t samples, const tensor_size_t count, rng_t& rng)
{
    auto udist = make_udist<tensor_size_t>(0, samples.size() - 1);

    auto selection = indices_t{count};
    std::generate(begin(selection), end(selection), [&]() { return samples(udist(rng)); });
    std::sort(begin(selection), end(selection));
    return selection;
}

indices_t nano::sample_with_replacement(sample_indices_t samples, const tensor_size_t count)
{
    auto rng = make_rng();
    return sample_with_replacement(samples, count, rng);
}

indices_t nano::sample_with_replacement(sample_indices_t samples, sample_weights_t weights, const tensor_size_t count,
                                        rng_t& rng)
{
    assert(weights.min() >= 0.0);
    assert(samples.size() == weights.size());

    auto wdist = std::discrete_distribution<tensor_size_t>(begin(weights), end(weights));

    auto selection = indices_t{count};
    std::generate(begin(selection), end(selection), [&]() { return samples(wdist(rng)); });
    std::sort(begin(selection), end(selection));
    return selection;
}

indices_t nano::sample_with_replacement(sample_indices_t samples, sample_weights_t weights, const tensor_size_t count)
{
    auto rng = make_rng();
    return sample_with_replacement(samples, weights, count, rng);
}

indices_t nano::sample_without_replacement(sample_indices_t samples_, const tensor_size_t count, rng_t& rng)
{
    assert(count <= samples_.size());

    auto samples = indices_t{samples_};
    std::shuffle(begin(samples), end(samples), rng);

    auto selection = samples.slice(0, count);
    std::sort(begin(selection), end(selection));
    return selection;
}

indices_t nano::sample_without_replacement(sample_indices_t samples, const tensor_size_t count)
{
    auto rng = make_rng();
    return sample_without_replacement(samples, count, rng);
}

matrix_t nano::sample_from_ball(const vector_t& x0, const scalar_t radius, const tensor_size_t count)
{
    auto rng = make_rng();
    return sample_from_ball(x0, radius, count, rng);
}

matrix_t nano::sample_from_ball(const vector_t& x0, const scalar_t radius, const tensor_size_t count, rng_t& rng)
{
    assert(count >= 0);
    assert(radius > 0.0);
    assert(x0.size() > 0);

    // see algorithm 4.1 from the reference, applied to the euclidean distance
    const auto n = x0.size();

    auto sign_dist    = std::discrete_distribution({1, 1});
    auto epsilon_dist = std::normal_distribution<scalar_t>{0.5, 2.0};
    auto scale_dist   = std::uniform_real_distribution<scalar_t>(0.0, 1.0);

    auto xs = matrix_t{count, n};
    for (tensor_size_t i = 0; i < count; ++i)
    {
        for (tensor_size_t k = 0; k < n; ++k)
        {
            xs(i, k) = epsilon_dist(rng) * (sign_dist(rng) == 0 ? -1.0 : +1.0);
        }

        const auto z = std::pow(scale_dist(rng), 1.0 / static_cast<scalar_t>(n));

        auto x    = xs.row(i).transpose();
        x.array() = x0.array() + radius * z * x.array() / x.lpNorm<2>();
    }

    return xs;
}
