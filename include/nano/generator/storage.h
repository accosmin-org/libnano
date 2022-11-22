#pragma once

#include <nano/scalar.h>
#include <nano/tensor/tensor.h>

namespace nano
{
    // single-label categorical feature values: (sample index) = label/class index
    // NB: negative values imply missing feature values.
    using sclass_mem_t  = tensor_mem_t<int32_t, 1>;
    using sclass_map_t  = tensor_map_t<int32_t, 1>;
    using sclass_cmap_t = tensor_cmap_t<int32_t, 1>;

    // multi-label categorical feature values: (sample index, label/class index) = 0 or 1
    // NB: any negative value imply missing feature values.
    using mclass_mem_t  = tensor_mem_t<int8_t, 2>;
    using mclass_map_t  = tensor_map_t<int8_t, 2>;
    using mclass_cmap_t = tensor_cmap_t<int8_t, 2>;

    // scalar continuous feature values: (sample index) = scalar feature value
    // NB: not-finite values imply missing feature values.
    using scalar_mem_t  = tensor_mem_t<scalar_t, 1>;
    using scalar_map_t  = tensor_map_t<scalar_t, 1>;
    using scalar_cmap_t = tensor_cmap_t<scalar_t, 1>;

    // structured continuous feature values: (sample index, dim1, dim2, dim3)
    // NB: any not-finite value imply missing feature values.
    using struct_mem_t  = tensor_mem_t<scalar_t, 4>;
    using struct_map_t  = tensor_map_t<scalar_t, 4>;
    using struct_cmap_t = tensor_cmap_t<scalar_t, 4>;

    // (original feature index, feature component, ...)
    using feature_mapping_t = tensor_mem_t<tensor_size_t, 2>;
} // namespace nano
