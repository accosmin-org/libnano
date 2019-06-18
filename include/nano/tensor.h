#pragma once

#include <nano/scalar.h>
#include <nano/tensor/tensor.h>

namespace nano
{
    using vector_t = tensor_vector_t<scalar_t>;
    using matrix_t = tensor_matrix_t<scalar_t>;

    using vector_map_t = Eigen::Map<vector_t>;
    using vector_cmap_t = Eigen::Map<const vector_t>;

    using tensor1d_t = tensor_mem_t<scalar_t, 1>;
    using tensor2d_t = tensor_mem_t<scalar_t, 2>;
    using tensor3d_t = tensor_mem_t<scalar_t, 3>;
    using tensor4d_t = tensor_mem_t<scalar_t, 4>;

    using tensor1ds_t = std::vector<tensor1d_t>;
    using tensor2ds_t = std::vector<tensor2d_t>;
    using tensor3ds_t = std::vector<tensor3d_t>;
    using tensor4ds_t = std::vector<tensor4d_t>;

    using tensor1d_dim_t = tensor1d_t::tdims;
    using tensor2d_dim_t = tensor2d_t::tdims;
    using tensor3d_dim_t = tensor3d_t::tdims;
    using tensor4d_dim_t = tensor4d_t::tdims;

    using tensor1d_dims_t = std::vector<tensor1d_dim_t>;
    using tensor2d_dims_t = std::vector<tensor2d_dim_t>;
    using tensor3d_dims_t = std::vector<tensor3d_dim_t>;
    using tensor4d_dims_t = std::vector<tensor4d_dim_t>;

    using tensor1d_map_t = tensor_map_t<scalar_t, 1>;
    using tensor2d_map_t = tensor_map_t<scalar_t, 2>;
    using tensor3d_map_t = tensor_map_t<scalar_t, 3>;
    using tensor4d_map_t = tensor_map_t<scalar_t, 4>;

    using tensor1d_maps_t = std::vector<tensor1d_map_t>;
    using tensor2d_maps_t = std::vector<tensor2d_map_t>;
    using tensor3d_maps_t = std::vector<tensor3d_map_t>;
    using tensor4d_maps_t = std::vector<tensor4d_map_t>;

    using tensor1d_cmap_t = tensor_cmap_t<scalar_t, 1>;
    using tensor2d_cmap_t = tensor_cmap_t<scalar_t, 2>;
    using tensor3d_cmap_t = tensor_cmap_t<scalar_t, 3>;
    using tensor4d_cmap_t = tensor_cmap_t<scalar_t, 4>;

    using tensor1d_cmaps_t = std::vector<tensor1d_cmap_t>;
    using tensor2d_cmaps_t = std::vector<tensor2d_cmap_t>;
    using tensor3d_cmaps_t = std::vector<tensor3d_cmap_t>;
    using tensor4d_cmaps_t = std::vector<tensor4d_cmap_t>;
}
