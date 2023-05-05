#pragma once

#include <nano/arch.h>
#include <nano/core/random.h>
#include <nano/scalar.h>
#include <nano/tensor/tensor.h>

namespace nano
{
using sample_indices_t = tensor_cmap_t<tensor_size_t, 1>;
using sample_weights_t = tensor_cmap_t<scalar_t, 1>;

///
/// \brief randomly sample with replacement `count` elements (aka bootstrapping).
///
/// NB: there may be duplicates in the returned indices.
/// NB: the returned indices are sorted to potentially improve speed.
///
NANO_PUBLIC indices_t sample_with_replacement(sample_indices_t, tensor_size_t count);
NANO_PUBLIC indices_t sample_with_replacement(sample_indices_t, tensor_size_t count, rng_t&);

///
/// \brief randomly sample with replacement `count` elements using the given per-sample weights (aka bootstrapping).
///
/// NB: there may be duplicates in the returned indices.
/// NB: the returned indices are sorted to potentially improve speed.
///
NANO_PUBLIC indices_t sample_with_replacement(sample_indices_t, sample_weights_t, tensor_size_t count);
NANO_PUBLIC indices_t sample_with_replacement(sample_indices_t, sample_weights_t, tensor_size_t count, rng_t&);

///
/// \brief randomly sample without replacement `count` elements.
///
/// NB: there won't be any duplicates in the returned indices.
/// NB: the returned indices are sorted to potentially improve speed.
///
NANO_PUBLIC indices_t sample_without_replacement(sample_indices_t, tensor_size_t count);
NANO_PUBLIC indices_t sample_without_replacement(sample_indices_t, tensor_size_t count, rng_t&);
} // namespace nano
