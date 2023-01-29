#pragma once

#include <algorithm>
#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief min-reduce the given set of accumulators (e.g. per thread) using the `m_score` attribute.
    ///
    template <typename taccumulator>
    const auto& min_reduce(const std::vector<taccumulator>& accumulators)
    {
        const auto op = [](const taccumulator& one, const taccumulator& other) { return one.m_score < other.m_score; };
        const auto it = std::min_element(accumulators.begin(), accumulators.end(), op);
        return *it;
    }

    ///
    /// \brief map-reduce the given set of accumulators (e.g. per thread) into the first accumulator.
    ///
    template <typename taccumulator>
    const auto& sum_reduce(std::vector<taccumulator>& accumulators, const tensor_size_t samples)
    {
        auto& accumulator0 = accumulators[0];
        for (size_t i = 1; i < accumulators.size(); ++i)
        {
            accumulator0 += accumulators[i];
        }
        return (accumulator0 /= samples);
    }
} // namespace nano
