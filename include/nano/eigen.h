#pragma once

#include <nano/scalar.h>
#include <nano/tensor/eigen.h>
#include <nano/tensor/index.h>

namespace nano
{
using vector_t      = tensor_vector_t<scalar_t>;
using vector_map_t  = Eigen::Map<vector_t>;
using vector_cmap_t = Eigen::Map<const vector_t>;

using matrix_t      = tensor_matrix_t<scalar_t>;
using matrix_map_t  = Eigen::Map<matrix_t>;
using matrix_cmap_t = Eigen::Map<const matrix_t>;
} // namespace nano
