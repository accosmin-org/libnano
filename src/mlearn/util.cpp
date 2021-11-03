#include <nano/core/random.h>
#include <nano/mlearn/util.h>

using namespace nano;

indices_t nano::sample_with_replacement(tensor_size_t samples, tensor_size_t count)
{
    auto rng = make_rng();
    auto udist = make_udist<tensor_size_t>(0, samples - 1);

    indices_t set(count);
    std::generate(begin(set), end(set), [&] () { return udist(rng); });
    std::sort(begin(set), end(set));

    return set;
}

indices_t nano::sample_without_replacement(tensor_size_t samples, tensor_size_t count)
{
    assert(0 <= samples && count <= samples);

    auto all = arange(0, samples);
    std::shuffle(begin(all), end(all), make_rng());

    indices_t set = all.slice(0, count);
    std::sort(begin(set), end(set));
    return set;
}
