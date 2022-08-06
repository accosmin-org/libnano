#include <algorithm>
#include <nano/core/random.h>

using namespace nano;

rng_t nano::make_rng(seed_t seed)
{
    if (seed)
    {
        auto rng = rng_t{}; // NOLINT(cert-msc32-c,cert-msc51-cpp)
        rng.seed(*seed);
        return rng;
    }
    else
    {
        static constexpr auto rng_state_bytes = rng_t::state_size * sizeof(rng_t::result_type);

        auto source = std::random_device{};

        std::random_device::result_type random_data[(rng_state_bytes - 1) / sizeof(source()) + 1];

        std::generate(std::begin(random_data), std::end(random_data), std::ref(source));

        auto seed_seq = std::seed_seq(std::begin(random_data), std::end(random_data));

        return rng_t{seed_seq};
    }
}
