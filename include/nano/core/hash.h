#pragma once

#include <cstdint>
#include <type_traits>

namespace nano::detail
{
    constexpr uint32_t hash_version()
    {
        return 0;
    }

    inline uint64_t hash_combine(const uint64_t seed, const uint64_t hash)
    {
        return seed ^ (hash + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }

    template <typename tscalar, typename tsize>
    uint64_t hash(const tscalar* data, const tsize size)
    {
        static_assert(sizeof(float) == 4);
        static_assert(sizeof(double) == 8);
        static_assert(sizeof(tscalar) <= 8);
        static_assert(std::is_arithmetic_v<tscalar>);

        uint64_t hash = 0;
        for (tsize i = 0; i < size; ++i)
        {
            if constexpr (std::is_floating_point_v<tscalar>)
            {
                if constexpr (sizeof(tscalar) == 4)
                {
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                    hash = hash_combine(hash, *reinterpret_cast<const uint32_t*>(&data[i]));
                }
                else
                {
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                    hash = hash_combine(hash, *reinterpret_cast<const uint64_t*>(&data[i]));
                }
            }
            else
            {
                hash = hash_combine(hash, static_cast<uint64_t>(data[i]));
            }
        }
        return hash;
    }
} // namespace nano::detail
