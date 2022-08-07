#include <algorithm>
#include <nano/core/random.h>

using namespace nano;

rng_t nano::make_rng(seed_t seed)
{
    if (seed)
    {
        return rng_t{static_cast<rng_t::result_type>(*seed)};
    }
    else
    {
        auto source = std::random_device{};
        return rng_t{static_cast<rng_t::result_type>(source())};
    }
}
