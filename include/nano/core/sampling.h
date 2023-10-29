#pragma once

#include <nano/arch.h>
#include <nano/core/random.h>
#include <nano/eigen.h>
#include <nano/tensor/tensor.h>

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
/// \brief uniformly sample `count` vectors `x` from the given n-dimensional ball: ||x - x0||_2 <= radius.
///
/// see "Uniform Sample Generation in lpBalls for Probabilistic Robustness Analysis", by Calafiore, Dabbene, Tempo, 1998
///
/// NB: the sampled vectors are stored in a matrix of shape (n_samples, vector_size).
///
NANO_PUBLIC matrix_t sample_from_ball(const vector_t& x0, const scalar_t radius, tensor_size_t count);
NANO_PUBLIC matrix_t sample_from_ball(const vector_t& x0, const scalar_t radius, tensor_size_t count, rng_t&);
} // namespace nano
