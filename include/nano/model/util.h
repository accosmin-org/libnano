#pragma once

#include <nano/arch.h>
#include <nano/core/seed.h>
#include <nano/tensor/tensor.h>

namespace nano
{
    ///
    /// \brief randomly sample with replacement `count` elements (aka bootstrapping).
    ///
    /// NB: there may be duplicates in the returned indices.
    /// NB: the returned indices are sorted to potentially improve speed.
    ///
    NANO_PUBLIC indices_t sample_with_replacement(tensor_size_t samples, tensor_size_t count, seed_t seed = seed_t{});
    NANO_PUBLIC indices_t sample_with_replacement(const indices_t& samples, tensor_size_t count,
                                                  seed_t seed = seed_t{});

    ///
    /// \brief randomly sample without replacement `count` elements.
    ///
    /// NB: there won't be any duplicates in the returned indices.
    /// NB: the returned indices are sorted to potentially improve speed.
    ///
    NANO_PUBLIC indices_t sample_without_replacement(tensor_size_t samples, tensor_size_t count,
                                                     seed_t seed = seed_t{});
    NANO_PUBLIC indices_t sample_without_replacement(const indices_t& samples, tensor_size_t count,
                                                     seed_t seed = seed_t{});
} // namespace nano
