#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
///
/// \brief construct consecutive tensor indices in the range [min, max).
///
inline auto arange(const tensor_size_t min, const tensor_size_t max)
{
    assert(min <= max);

    indices_t indices(max - min);
    indices.lin_spaced(min, max - 1);
    return indices;
}

///
/// \brief create a tensor from an initializer list.
///
template <typename tscalar, size_t trank, typename... tvalues>
auto make_tensor(const tensor_dims_t<trank>& dims, tvalues... values)
{
    const auto list = {static_cast<tscalar>(values)...};
    assert(::nano::size(dims) == static_cast<tensor_size_t>(list.size()));

    tensor_mem_t<tscalar, trank> tensor{dims};
    std::copy(list.begin(), list.end(), begin(tensor));
    return tensor;
}

///
/// \brief create indices from an initializer list.
///
template <typename... tindices>
auto make_indices(tindices... indices)
{
    return make_tensor<tensor_size_t>(make_dims(static_cast<tensor_size_t>(sizeof...(indices))), indices...);
}

///
/// \brief create a tensor and fill it with the given value.
///
template <typename tscalar, size_t trank, typename tscalar_value>
auto make_full_tensor(const tensor_dims_t<trank>& dims, tscalar_value value)
{
    tensor_mem_t<tscalar, trank> tensor(dims);
    tensor.full(static_cast<tscalar>(value));
    return tensor;
}

///
/// \brief create a tensor and fill it with random values.
///
template <typename tscalar, size_t trank, typename tscalar_value = tscalar>
auto make_random_tensor(const tensor_dims_t<trank>& dims, const tscalar_value min_value = -1,
                        const tscalar_value max_value = +1, const seed_t seed = seed_t{})
{
    tensor_mem_t<tscalar, trank> tensor(dims);
    tensor.random(static_cast<tscalar>(min_value), static_cast<tscalar>(max_value), seed);
    return tensor;
}

///
/// \brief create a matrix from an initializer list.
///
template <typename tscalar, typename... tvalues>
auto make_matrix(const tensor_size_t rows, tvalues... values)
{
    const auto list = {static_cast<tscalar>(values)...};
    const auto size = static_cast<tensor_size_t>(list.size());
    assert(size % rows == 0);

    tensor_mem_t<tscalar, 2> matrix{rows, size / rows};
    std::copy(list.begin(), list.end(), begin(matrix));
    return matrix;
}

///
/// \brief create a vector from an initializer list.
///
template <typename tscalar, typename... tvalues>
auto make_vector(tvalues... values)
{
    const auto list = {static_cast<tscalar>(values)...};

    tensor_mem_t<tscalar, 1> vector{static_cast<tensor_size_t>(list.size())};
    std::copy(list.begin(), list.end(), begin(vector));
    return vector;
}

///
/// \brief create a vector and fill it with the given value.
///
template <typename tscalar, typename tscalar_value>
auto make_full_vector(const tensor_size_t rows, const tscalar_value value)
{
    return make_full_tensor<tscalar>(make_dims(rows), value);
}

///
/// \brief create a vector and fill it with random values uniformly distributed in the given range.
///
template <typename tscalar, typename tscalar_value = tscalar>
auto make_random_vector(const tensor_size_t rows, const tscalar_value min_value = -1,
                        const tscalar_value max_value = +1, const seed_t seed = seed_t{})
{
    return make_random_tensor<tscalar>(make_dims(rows), min_value, max_value, seed);
}

///
/// \brief create a vector and fill it with the given value.
///
template <typename tscalar, typename tscalar_value = tscalar>
auto make_full_matrix(const tensor_size_t rows, const tensor_size_t cols, const tscalar_value value)
{
    return make_full_tensor<tscalar>(make_dims(rows, cols), value);
}

///
/// \brief create a matrix and fill it with random values uniformly distributed in the given range.
///
template <typename tscalar, typename tscalar_value = tscalar>
auto make_random_matrix(const tensor_size_t rows, const tensor_size_t cols, const tscalar_value min_value = -1,
                        const tscalar_value max_value = +1, const seed_t seed = seed_t{})
{
    return make_random_tensor<tscalar>(make_dims(rows, cols), min_value, max_value, seed);
}
} // namespace nano
