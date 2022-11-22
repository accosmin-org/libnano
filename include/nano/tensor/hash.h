#pragma once

#include <functional>
#include <nano/tensor/index.h>

namespace nano::detail
{
    inline constexpr uint32_t tensor_version()
    {
        return 0;
    }

    inline uint64_t hash_combine(const uint64_t seed, const uint64_t hash)
    {
        return seed ^ (hash + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    }

    template <typename tscalar>
    uint64_t hash(const tscalar* data, const tensor_size_t size)
    {
        const auto hasher = std::hash<tscalar>{};

        uint64_t hash = 0;
        for (tensor_size_t i = 0; i < size; ++i)
        {
            hash = hash_combine(hash, hasher(data[i]));
        }
        return hash;
    }
} // namespace nano::detail
