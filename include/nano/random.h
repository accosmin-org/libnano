#pragma once

#include <random>
#include <type_traits>

namespace nano
{
    using rng_t = std::minstd_rand;

    template <typename tscalar, typename = typename std::is_arithmetic<tscalar>::type>
    using udist_t = typename std::conditional<
        std::is_integral<tscalar>::value,
        std::uniform_int_distribution<tscalar>,
        std::uniform_real_distribution<tscalar>>::type;

    ///
    /// \brief create & initialize a random number generator.
    ///
    inline auto make_rng()
    {
        // todo: use seed_seq to initialize the RNG (see C++17)
        return rng_t{std::random_device{}()};
    }

    ///
    /// \brief create an uniform distribution for the [min, max] range.
    ///
    template <typename tscalar, typename = typename std::is_arithmetic<tscalar>::type>
    inline auto make_udist(const tscalar min, const tscalar max)
    {
        assert(min <= max);
        return udist_t<tscalar>(min, max);
    }

    ///
    /// \brief generate a random value uniformaly distributed in the [min, max] range.
    ///
    template <typename tscalar, typename trng, typename = typename std::is_arithmetic<tscalar>::type>
    tscalar urand(const tscalar min, const tscalar max, trng&& rng)
    {
        auto udist = make_udist<tscalar>(min, max);
        return udist(rng);
    }

    ///
    /// \brief fill the [begin, range) range of elements with random values uniformaly distributed in the [min, max] range.
    ///
    template <typename tscalar, typename titerator, typename trng, typename = typename std::is_arithmetic<tscalar>::type>
    void urand(const tscalar min, const tscalar max, titerator begin, const titerator end, trng&& rng)
    {
        auto udist = make_udist<tscalar>(min, max);
        for ( ; begin != end; ++ begin)
        {
            *begin = udist(rng);
        }
    }

    ///
    /// \brief add to the [begin, range) range of elements random values uniformaly distributed in the [min, max] range.
    ///
    template <typename tscalar, typename titerator, typename trng, typename = typename std::is_arithmetic<tscalar>::type>
    void add_urand(const tscalar min, const tscalar max, titerator begin, const titerator end, trng&& rng)
    {
        auto udist = make_udist<tscalar>(min, max);
        for ( ; begin != end; ++ begin)
        {
            *begin += udist(rng);
        }
    }
}
