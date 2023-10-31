#pragma once

#include <nano/scalar.h>
#include <nano/tensor/makers.h>
#include <nano/tensor/operators.h>
#include <nano/tensor/traits.h>

namespace nano
{
using vector_t      = tensor_mem_t<scalar_t, 1>;
using vector_map_t  = tensor_map_t<scalar_t, 1>;
using vector_cmap_t = tensor_cmap_t<scalar_t, 1>;

using matrix_t      = tensor_mem_t<scalar_t, 2>;
using matrix_map_t  = tensor_map_t<scalar_t, 2>;
using matrix_cmap_t = tensor_cmap_t<scalar_t, 2>;

using tensor1d_t = tensor_mem_t<scalar_t, 1>;
using tensor2d_t = tensor_mem_t<scalar_t, 2>;
using tensor3d_t = tensor_mem_t<scalar_t, 3>;
using tensor4d_t = tensor_mem_t<scalar_t, 4>;
using tensor5d_t = tensor_mem_t<scalar_t, 5>;

using tensor1d_dims_t = tensor1d_t::tdims;
using tensor2d_dims_t = tensor2d_t::tdims;
using tensor3d_dims_t = tensor3d_t::tdims;
using tensor4d_dims_t = tensor4d_t::tdims;
using tensor5d_dims_t = tensor5d_t::tdims;

using tensor1d_map_t = tensor_map_t<scalar_t, 1>;
using tensor2d_map_t = tensor_map_t<scalar_t, 2>;
using tensor3d_map_t = tensor_map_t<scalar_t, 3>;
using tensor4d_map_t = tensor_map_t<scalar_t, 4>;
using tensor5d_map_t = tensor_map_t<scalar_t, 5>;

using tensor1d_cmap_t = tensor_cmap_t<scalar_t, 1>;
using tensor2d_cmap_t = tensor_cmap_t<scalar_t, 2>;
using tensor3d_cmap_t = tensor_cmap_t<scalar_t, 3>;
using tensor4d_cmap_t = tensor_cmap_t<scalar_t, 4>;
using tensor5d_cmap_t = tensor_cmap_t<scalar_t, 5>;
} // namespace nano
