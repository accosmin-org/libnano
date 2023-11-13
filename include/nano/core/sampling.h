#pragma once

#include <nano/arch.h>
#include <nano/core/random.h>
#include <nano/tensor.h>

namespace nano
{
using sample_indices_t = tensor_cmap_t<tensor_size_t, 1>;
using sample_weights_t = tensor_cmap_t<scalar_t, 1>;

///
/// \brief uniformly sample with replacement `count` elements (aka bootstrapping).
///
/// NB: there may be duplicates in the returned indices.
/// NB: the returned indices are sorted to potentially improve speed.
///
NANO_PUBLIC indices_t sample_with_replacement(sample_indices_t, tensor_size_t count);
NANO_PUBLIC indices_t sample_with_replacement(sample_indices_t, tensor_size_t count, rng_t&);

///
/// \brief uniformly sample with replacement `count` elements using the given per-sample weights (aka bootstrapping).
///
/// NB: there may be duplicates in the returned indices.
/// NB: the returned indices are sorted to potentially improve speed.
///
NANO_PUBLIC indices_t sample_with_replacement(sample_indices_t, sample_weights_t, tensor_size_t count);
NANO_PUBLIC indices_t sample_with_replacement(sample_indices_t, sample_weights_t, tensor_size_t count, rng_t&);

///
/// \brief uniformly sample without replacement `count` elements.
///
/// NB: there won't be any duplicates in the returned indices.
/// NB: the returned indices are sorted to potentially improve speed.
///
NANO_PUBLIC indices_t sample_without_replacement(sample_indices_t, tensor_size_t count);
NANO_PUBLIC indices_t sample_without_replacement(sample_indices_t, tensor_size_t count, rng_t&);

///
/// \brief uniformly sample a vector x from the given n-dimensional ball: ||x - x0||_2 <= radius.
///
/// see "Uniform Sample Generation in lpBalls for Probabilistic Robustness Analysis", by Calafiore, Dabbene, Tempo, 1998
///
NANO_PUBLIC vector_t sample_from_ball(vector_cmap_t x0, scalar_t radius);
NANO_PUBLIC vector_t sample_from_ball(vector_cmap_t x0, scalar_t radius, rng_t&);

NANO_PUBLIC void sample_from_ball(vector_cmap_t x0, scalar_t radius, vector_map_t x);
NANO_PUBLIC void sample_from_ball(vector_cmap_t x0, scalar_t radius, vector_map_t x, rng_t&);
} // namespace nano
