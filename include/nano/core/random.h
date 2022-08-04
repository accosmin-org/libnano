#pragma once

#include <algorithm>
#include <cassert>
#include <nano/core/seed.h>
#include <random>
#include <type_traits>

namespace nano
{
    using rng_t = std::mt19937_64;

    template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
    using udist_t = typename std::conditional<std::is_integral_v<tscalar>, std::uniform_int_distribution<tscalar>,
                                              std::uniform_real_distribution<tscalar>>::type;

    ///
    /// \brief create & initialize a random number generator.
    ///
    inline auto make_rng(seed_t seed = seed_t{})
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

            auto seed_seq = std::seed_seq{std::begin(random_data), std::end(random_data)};

            return rng_t{seed_seq};
        }
    }

    ///
    /// \brief create an uniform distribution for the [min, max] range.
    ///
    template <typename tscalar, std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
    inline auto make_udist(tscalar min, tscalar max)
    {
        assert(min <= max);
        return udist_t<tscalar>(min, max);
    }

    ///
    /// \brief generate a random value uniformaly distributed in the [min, max] range.
    ///
    template <typename tscalar, typename trng,
              std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
    tscalar urand(tscalar min, tscalar max, trng&& rng)
    {
        auto udist = make_udist<tscalar>(min, max);
        return udist(rng);
    }

    ///
    /// \brief fill the [begin, range) range of elements with random values uniformaly distributed in the [min, max]
    /// range.
    ///
    template <typename tscalar, typename titerator, typename trng,
              std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
    void urand(tscalar min, tscalar max, titerator begin, const titerator end, trng&& rng)
    {
        auto udist = make_udist<tscalar>(min, max);
        for (; begin != end; ++begin)
        {
            *begin = udist(rng);
        }
    }
} // namespace nano
