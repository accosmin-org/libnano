#include <nano/core/random.h>
#include <nano/model/util.h>

using namespace nano;

template <typename toperator>
static auto sample_with_replacement(const tensor_size_t samples, const tensor_size_t count, const seed_t seed,
                                    const toperator& op)
{
    auto rng   = make_rng(seed);
    auto udist = make_udist<tensor_size_t>(0, samples - 1);

    auto selection = indices_t{count};
    std::generate(begin(selection), end(selection), [&]() { return op(udist(rng)); });
    std::sort(begin(selection), end(selection));
    return selection;
}

static auto sample_without_replacement(indices_t samples, const tensor_size_t count, const seed_t seed)
{
    assert(count <= samples.size());

    std::shuffle(begin(samples), end(samples), make_rng(seed));

    auto selection = samples.slice(0, count);
    std::sort(begin(selection), end(selection));
    return selection;
}

indices_t nano::sample_with_replacement(const indices_t& samples, const tensor_size_t count, const seed_t seed)
{
    return ::sample_with_replacement(samples.size(), count, seed, [&](const auto index) { return samples(index); });
}

indices_t nano::sample_with_replacement(const tensor_size_t samples, const tensor_size_t count, const seed_t seed)
{
    return ::sample_with_replacement(samples, count, seed, [](const auto index) { return index; });
}

indices_t nano::sample_without_replacement(const tensor_size_t samples, const tensor_size_t count, const seed_t seed)
{
    return ::sample_without_replacement(arange(0, samples), count, seed);
}

indices_t nano::sample_without_replacement(const indices_t& samples, const tensor_size_t count, const seed_t seed)
{
    return ::sample_without_replacement(samples, count, seed);
}
