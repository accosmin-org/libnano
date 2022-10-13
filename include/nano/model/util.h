#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief randomly sample with replacement `count` elements from the given total number of samples.
    ///
    /// NB: there may be duplicates in the returned indices.
    /// NB: the returned indices in the range [0, samples) are sorted to potentially improve speed.
    ///
    NANO_PUBLIC indices_t sample_with_replacement(tensor_size_t samples, tensor_size_t count);

    ///
    /// \brief randomly sample without replacement `count` elements from the given total number of samples.
    ///
    /// NB: there won't be any duplicates in the returned indices.
    /// NB: the returned indices in the range [0, samples) are sorted to potentially improve speed.
    ///
    NANO_PUBLIC indices_t sample_without_replacement(tensor_size_t samples, tensor_size_t count);
} // namespace nano
