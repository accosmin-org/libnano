#include <nano/core/sampling.h>

using namespace nano;

indices_t nano::sample_with_replacement(const indices_t& samples, const tensor_size_t count, rng_t& rng)
{
    auto udist = make_udist<tensor_size_t>(0, samples.size() - 1);

    auto selection = indices_t{count};
    std::generate(begin(selection), end(selection), [&]() { return samples(udist(rng)); });
    std::sort(begin(selection), end(selection));
    return selection;
}

indices_t nano::sample_with_replacement(const indices_t& samples, const tensor_size_t count)
{
    auto rng = make_rng();
    return sample_with_replacement(samples, count, rng);
}

indices_t nano::sample_without_replacement(const indices_t& samples_, const tensor_size_t count, rng_t& rng)
{
    assert(count <= samples_.size());

    auto samples = samples_;
    std::shuffle(begin(samples), end(samples), rng);

    auto selection = samples.slice(0, count);
    std::sort(begin(selection), end(selection));
    return selection;
}

indices_t nano::sample_without_replacement(const indices_t& samples, const tensor_size_t count)
{
    auto rng = make_rng();
    return sample_without_replacement(samples, count, rng);
}
