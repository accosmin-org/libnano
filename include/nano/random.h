#pragma once

#include <random>
#include <cassert>
#include <numeric>
#include <algorithm>
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

    ///
    /// \brief randomly split count elements in percentage_value1% having value1 and the rest value2.
    ///
    template <typename tvalue>
    auto split2(const size_t count,
        const tvalue value1, const size_t percentage_value1,
        const tvalue value2)
    {
        assert(percentage_value1 <= 100);

        const auto begin0 = size_t(0);
        const auto begin1 = begin0 + count * percentage_value1 / 100;
        const auto begin2 = count;

        std::vector<tvalue> values(count);
        std::fill(values.begin() + begin0, values.begin() + begin1, value1);
        std::fill(values.begin() + begin1, values.begin() + begin2, value2);

        std::shuffle(values.begin(), values.end(), make_rng());
        return values;
    }

    ///
    /// \brief randomly split count elements in percentage_value1% having value1,
    ///     percentage_value2% having value2 and the rest value3.
    ///
    template <typename tvalue>
    auto split3(const size_t count,
        const tvalue value1, const size_t percentage_value1,
        const tvalue value2, const size_t percentage_value2,
        const tvalue value3)
    {
        assert(percentage_value1 + percentage_value2 <= 100);

        const auto begin0 = size_t(0);
        const auto begin1 = begin0 + count * percentage_value1 / 100;
        const auto begin2 = begin1 + count * percentage_value2 / 100;
        const auto begin3 = count;

        std::vector<tvalue> values(count);
        std::fill(values.begin() + begin0, values.begin() + begin1, value1);
        std::fill(values.begin() + begin1, values.begin() + begin2, value2);
        std::fill(values.begin() + begin2, values.begin() + begin3, value3);

        std::shuffle(values.begin(), values.end(), make_rng());
        return values;
    }

    ///
    /// \brief sample with replacement the given percentage of `size` elements.
    ///
    template <typename tsize, typename tpercentage>
    std::vector<tsize> sample_with_replacement(const tsize size, const tpercentage percentage)
    {
        assert(0 <= percentage && percentage <= 100);

        auto rng = make_rng();
        auto udist = make_udist<tsize>(tsize(0), size - 1);

        std::vector<tsize> indices(static_cast<size_t>(percentage * size / 100));
        std::generate(indices.begin(), indices.end(), [&] () { return udist(rng); });

        std::sort(indices.begin(), indices.end());
        return indices;
    }

    ///
    /// \brief sample without replacement the given percentage of `size` elements.
    ///
    template <typename tsize, typename tpercentage>
    std::vector<tsize> sample_without_replacement(const tsize size, const tpercentage percentage)
    {
        assert(0 <= percentage && percentage <= 100);

        std::vector<tsize> indices(size);
        std::iota(indices.begin(), indices.end(), tsize(0));

        std::shuffle(indices.begin(), indices.end(), make_rng());

        std::vector<tsize> ret{indices.begin(), indices.begin() + (percentage * size / 100)};
        std::sort(ret.begin(), ret.end());
        return ret;
    }
}
