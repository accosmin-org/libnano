#pragma once

#include <cassert>
#include <nano/arch.h>
#include <nano/core/seed.h>
#include <random>
#include <type_traits>

namespace nano
{
using rng_t = std::minstd_rand;

template <class tscalar, std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
using udist_t = typename std::conditional_t<std::is_integral_v<tscalar>, std::uniform_int_distribution<tscalar>,
                                            std::uniform_real_distribution<tscalar>>;

///
/// \brief create & initialize a random number generator.
///
NANO_PUBLIC rng_t make_rng(seed_t seed = seed_t{});

///
/// \brief create an uniform distribution for the [min, max] range.
///
template <class tscalar, std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
inline auto make_udist(const tscalar min, const tscalar max)
{
    assert(min <= max);
    if constexpr (std::is_floating_point_v<tscalar>)
    {
        return udist_t<tscalar>(min, max);
    }
    else if constexpr (std::is_unsigned_v<tscalar>)
    {
        return udist_t<uint64_t>(min, max);
    }
    else
    {
        return udist_t<int64_t>(min, max);
    }
}

///
/// \brief generate a random value uniformaly distributed in the [min, max] range.
///
template <class tscalar, std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
tscalar urand(const tscalar min, const tscalar max, rng_t& rng)
{
    auto udist = make_udist<tscalar>(min, max);
    return static_cast<tscalar>(udist(rng));
}

template <class tscalar, std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
tscalar urand(const tscalar min, const tscalar max, const seed_t seed = seed_t{})
{
    auto rng = make_rng(seed);
    return urand(min, max, rng);
}

///
/// \brief fill the [begin, range) range of elements with random values uniformaly distributed in the [min, max]
/// range.
///
template <class tscalar, class titerator,
          std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
void urand(const tscalar min, const tscalar max, titerator begin, const titerator end, rng_t& rng)
{
    auto udist = make_udist<tscalar>(min, max);
    for (; begin != end; ++begin)
    {
        *begin = static_cast<tscalar>(udist(rng));
    }
}

template <class tscalar, class titerator,
          std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<tscalar>>, bool> = true>
void urand(const tscalar min, const tscalar max, titerator begin, const titerator end, const seed_t seed = seed_t{})
{
    auto rng = make_rng(seed);
    return urand(min, max, begin, end, rng);
}
} // namespace nano
