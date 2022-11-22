#pragma once

#include <nano/generator/storage.h>
#include <nano/tensor/hash.h>

namespace nano
{
    using mhashes_t      = tensor_mem_t<uint64_t, 1>;
    using mhashes_cmap_t = tensor_cmap_t<uint64_t, 1>;

    NANO_PUBLIC mhashes_t make_mhashes(const mclass_cmap_t& fvalues);

    template <typename tfvalues>
    uint64_t mhash(const tfvalues& sfvalues)
    {
        return ::nano::detail::hash(sfvalues.data(), sfvalues.size());
    }

    template <typename tfvalues>
    tensor_size_t find(const mhashes_t& mhashes, const tfvalues& values)
    {
        const auto* const begin = mhashes.begin();
        const auto* const end   = mhashes.end();

        const auto value = ::nano::mhash(values);

        const auto it = std::lower_bound(begin, end, value);
        return (it == end || *it != value) ? tensor_size_t{-1} : static_cast<tensor_size_t>(std::distance(begin, it));
    }
} // namespace nano
