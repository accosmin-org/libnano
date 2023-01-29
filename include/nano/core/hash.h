#pragma once

#include <cstdint>
#include <functional>

namespace nano::detail
{
    inline constexpr uint32_t hash_version()
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
        const auto hasher = std::hash<tscalar>{};

        uint64_t hash = 0;
        for (tsize i = 0; i < size; ++i)
        {
            hash = hash_combine(hash, hasher(data[i]));
        }
        return hash;
    }
} // namespace nano::detail
