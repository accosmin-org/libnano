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
