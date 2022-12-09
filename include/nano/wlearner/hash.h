#pragma once

#include <nano/generator/storage.h>
#include <nano/tensor/hash.h>

namespace nano::wlearner
{
    using hashes_t      = tensor_mem_t<uint64_t, 1>;
    using hashes_cmap_t = tensor_cmap_t<uint64_t, 1>;

    NANO_PUBLIC hashes_t make_hashes(const sclass_cmap_t& values);
    NANO_PUBLIC hashes_t make_hashes(const mclass_cmap_t& values);

    template <typename tfvalues>
    uint64_t hash(const tfvalues& values)
    {
        if constexpr (std::is_arithmetic_v<tfvalues>)
        {
            // single-label case
            return static_cast<uint64_t>(values);
        }
        else
        {
            // multi-label case
            return ::nano::detail::hash(values.data(), values.size());
        }
    }

    template <typename tfvalues>
    tensor_size_t find(const hashes_t& hashes, const tfvalues& values)
    {
        const auto hash = ::nano::wlearner::hash(values);

        const auto* const begin = hashes.begin();
        const auto* const end   = hashes.end();

        const auto it = std::lower_bound(begin, end, hash);
        return (it == end || *it != hash) ? tensor_size_t{-1} : static_cast<tensor_size_t>(std::distance(begin, it));
    }
} // namespace nano::wlearner
