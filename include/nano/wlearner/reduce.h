#pragma once

#include <algorithm>
#include <nano/tensor.h>

namespace nano::wlearner
{
    ///
    /// \brief min-reduce the given set of per-thread caches using the `min_score` attribute.
    ///
    template <typename tcache>
    const auto& min_reduce(const std::vector<tcache>& caches)
    {
        const auto op = [](const tcache& one, const tcache& other) { return one.m_score < other.m_score; };
        const auto it = std::min_element(caches.begin(), caches.end(), op);
        return *it;
    }

    ///
    /// \brief map-reduce the given set of per-thread caches into the first cache.
    ///
    template <typename tcache>
    const auto& sum_reduce(std::vector<tcache>& caches, const tensor_size_t samples)
    {
        auto& cache0 = caches[0];
        for (size_t i = 1; i < caches.size(); ++i)
        {
            cache0 += caches[i];
        }
        return (cache0 /= samples);
    }
} // namespace nano::wlearner
