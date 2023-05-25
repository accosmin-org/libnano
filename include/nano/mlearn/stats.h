#pragma once

#include <nano/tensor.h>

namespace nano::ml
{
///
/// \brief statistics related to evaluating a machine learning model
///     (e.g. using the loss function or the error metric).
///
struct stats_t
{
    scalar_t m_mean{0.0};
    scalar_t m_stdev{0.0};
    scalar_t m_count{0.0};
    scalar_t m_per01{0.0};
    scalar_t m_per05{0.0};
    scalar_t m_per10{0.0};
    scalar_t m_per20{0.0};
    scalar_t m_per50{0.0};
    scalar_t m_per80{0.0};
    scalar_t m_per90{0.0};
    scalar_t m_per95{0.0};
    scalar_t m_per99{0.0};
};

///
/// \brief compute statistics and store them in the given buffer.
///
NANO_PUBLIC void store_stats(const tensor1d_map_t& values, const tensor1d_map_t& stats);

///
/// \brief load the computed statistics from the given buffer.
///
NANO_PUBLIC stats_t load_stats(const tensor1d_cmap_t& stats);
} // namespace nano::ml
