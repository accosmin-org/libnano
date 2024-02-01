#pragma once

#include <nano/core/random.h>
#include <nano/tensor/numeric.h>
#include <nano/tensor/pprint.h>
#include <nano/tensor/range.h>
#include <nano/tensor/storage.h>
#include <nano/tensor/traits.h>

namespace nano
{
template <template <typename, size_t> class, typename, size_t>
class tensor_t;

///
/// \brief tensor that owns the allocated memory.
///
template <typename tscalar, size_t trank>
using tensor_mem_t = tensor_t<tensor_vector_storage_t, tscalar, trank>;

///
/// \brief tensor mapping a non-constant array.
///
template <typename tscalar, size_t trank>
using tensor_map_t = tensor_t<tensor_marray_storage_t, tscalar, trank>;

///
/// \brief tensor mapping a constant array.
///
template <typename tscalar, size_t trank>
using tensor_cmap_t = tensor_t<tensor_carray_storage_t, tscalar, trank>;

///
/// \brief tensor indices.
///
using indices_t      = tensor_mem_t<tensor_size_t, 1>;
using indices_map_t  = tensor_map_t<tensor_size_t, 1>;
using indices_cmap_t = tensor_cmap_t<tensor_size_t, 1>;

///
/// \brief map non-constant data to tensors.
///
template <typename tscalar, size_t trank>
auto map_tensor(tscalar* data, const tensor_dims_t<trank>& dims)
{
    return tensor_map_t<tscalar, trank>(data, dims);
}

///
/// \brief map constant data to tensors.
///
template <typename tscalar, size_t trank>
auto map_tensor(const tscalar* data, const tensor_dims_t<trank>& dims)
{
    return tensor_cmap_t<tscalar, trank>(data, dims);
}

///
/// \brief map non-constant data to tensors.
///
template <typename tscalar, typename... tsizes>
auto map_tensor(tscalar* data, tsizes... dims)
{
    return map_tensor(data, make_dims(dims...));
}

///
/// \brief map constant data to tensors.
///
template <typename tscalar, typename... tsizes>
auto map_tensor(const tscalar* data, tsizes... dims)
{
    return map_tensor(data, make_dims(dims...));
}

///
/// \brief return the default minimum range for random sampling of tensor values.
///
template <typename tscalar>
static constexpr auto default_min_random()
{
    if constexpr (std::is_unsigned_v<tscalar>)
    {
        return tscalar{0};
    }
    else
    {
        return tscalar{-1};
    }
}

///
/// \brief tensor w/o owning the allocated continuous memory.
///
/// NB: all access operations (e.g. Eigen arrays, vectors or matrices or sub-tensors) are performed
///     using only continuous memory.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
class tensor_t : public tstorage<tscalar, trank>
{
public:
    using tbase = tstorage<tscalar, trank>;

    using tbase::cols;
    using tbase::data;
    using tbase::dims;
    using tbase::offset;
    using tbase::offset0;
    using tbase::rank;
    using tbase::resizable;
    using tbase::resize;
    using tbase::rows;
    using tbase::size;
    using tdims = typename tbase::tdims;

    ///
    /// \brief default constructor
    ///
    tensor_t() = default;

    ///
    /// \brief construct to match the given size.
    ///
    template <typename... tsizes>
    explicit tensor_t(tsizes... dimensions)
        : tbase(make_dims(dimensions...))
    {
    }

    explicit tensor_t(tdims dimensions)
        : tbase(std::move(dimensions))
    {
    }

    ///
    /// \brief map non-ownning mutable C-arrays.
    ///
    explicit tensor_t(tscalar* ptr, tdims dimensions)
        : tbase(ptr, std::move(dimensions))
    {
    }

    ///
    /// \brief map non-ownning constant C-arrays.
    ///
    explicit tensor_t(const tscalar* ptr, tdims dimensions)
        : tbase(ptr, std::move(dimensions))
    {
    }

    ///
    /// \brief construct from an Eigen expression.
    ///
    template <typename texpression, std::enable_if_t<is_eigen_v<texpression>, bool> = true>
    // cppcheck-suppress noExplicitConstructor
    tensor_t(const texpression& expression) // NOLINT(hicpp-explicit-conversions)
    {
        static_assert(resizable);
        assign(expression);
    }

    ///
    /// \brief enable copying (delegate to the storage object).
    ///
    tensor_t(const tensor_t&)            = default;
    tensor_t& operator=(const tensor_t&) = default;

    ///
    /// \brief enable moving (delegate to the storage object).
    ///
    tensor_t(tensor_t&&) noexcept            = default;
    tensor_t& operator=(tensor_t&&) noexcept = default;

    ///
    /// \brief copy constructor from different types (e.g. const array from mutable array).
    ///
    template <template <typename, size_t> class tstorage2>
    // cppcheck-suppress noExplicitConstructor
    tensor_t(const tensor_t<tstorage2, tscalar, trank>& other) // NOLINT(hicpp-explicit-conversions)
        : tbase(static_cast<const tstorage2<tscalar, trank>&>(other))
    {
    }

    ///
    /// \brief copy constructor from different types (e.g. mutable array from mutable vector).
    ///
    template <template <typename, size_t> class tstorage2>
    // cppcheck-suppress noExplicitConstructor
    tensor_t(tensor_t<tstorage2, tscalar, trank>& other) // NOLINT(hicpp-explicit-conversions)
        : tbase(static_cast<tstorage2<tscalar, trank>&>(other))
    {
    }

    ///
    /// \brief assignment operator from different types (e.g. const from non-const scalars).
    ///
    template <template <typename, size_t> class tstorage2>
    tensor_t& operator=(const tensor_t<tstorage2, tscalar, trank>& other)
    {
        tbase::operator=(other);
        return *this;
    }

    ///
    /// \brief assignment operator from an Eigen expression.
    ///
    template <typename texpression, std::enable_if_t<is_eigen_v<texpression>, bool> = true>
    tensor_t& operator=(const texpression& expression)
    {
        assign(expression);
        return *this;
    }

    ///
    /// \brief default destructor
    ///
    ~tensor_t() = default;

    ///
    /// \brief access a continuous part of the tensor as an Eigen vector
    ///     (assuming the last dimensions that are ignored are zero).
    ///
    template <typename... tindices>
    auto vector(const tindices... indices)
    {
        return tvector(data(), indices...);
    }

    template <typename... tindices>
    auto vector(const tindices... indices) const
    {
        return tvector(data(), indices...);
    }

    ///
    /// \brief access a part of the tensor as an Eigen array
    ///     (assuming the last dimensions that are ignored are zero).
    ///
    template <typename... tindices>
    auto array(const tindices... indices)
    {
        return vector(indices...).array();
    }

    template <typename... tindices>
    auto array(const tindices... indices) const
    {
        return vector(indices...).array();
    }

    ///
    /// \brief access a part of the tensor as an Eigen matrix
    ///     (assuming that the last two dimensions are ignored).
    ///
    template <typename... tindices>
    auto matrix(const tindices... indices)
    {
        return tmatrix(data(), indices...);
    }

    template <typename... tindices>
    auto matrix(const tindices... indices) const
    {
        return tmatrix(data(), indices...);
    }

    ///
    /// \brief access a part of the tensor as a (sub-)tensor
    ///     (assuming the last dimensions that are ignored are zero)
    ///
    template <typename... tindices>
    auto tensor(const tindices... indices)
    {
        return ttensor(data(), indices...);
    }

    template <typename... tindices>
    auto tensor(const tindices... indices) const
    {
        return ttensor(data(), indices...);
    }

    ///
    /// \brief access a part of the tensor as a (sub-)tensor
    ///     (by taking the [begin, end) range of the first dimension)
    ///
    auto slice(const tensor_size_t begin, const tensor_size_t end) { return tslice(data(), begin, end); }

    auto slice(const tensor_size_t begin, const tensor_size_t end) const { return tslice(data(), begin, end); }

    auto slice(const tensor_range_t& range) { return tslice(data(), range.begin(), range.end()); }

    auto slice(const tensor_range_t& range) const { return tslice(data(), range.begin(), range.end()); }

    ///
    /// \brief copy some of (sub-)tensors using the given indices.
    /// NB: the indices are relative to the first dimension.
    ///
    template <typename tscalar_return = tscalar>
    void indexed(indices_cmap_t indices, tensor_mem_t<tscalar_return, trank>& subtensor) const
    {
        auto dimensions = dims();
        dimensions[0]   = indices.size();
        subtensor.resize(dimensions);

        indexed(indices, subtensor.tensor());
    }

    template <typename tscalar_return = tscalar>
    void indexed(indices_cmap_t indices, tensor_map_t<tscalar_return, trank> subtensor) const
    {
        assert(indices.min() >= 0 && indices.max() < this->template size<0>());

        auto dimensions = dims();
        dimensions[0]   = indices.size();
        assert(subtensor.dims() == dimensions);

        if constexpr (trank > 1)
        {
            for (tensor_size_t i = 0, indices_size = indices.size(); i < indices_size; ++i)
            {
                subtensor.vector(i) = vector(indices(i)).template cast<tscalar_return>();
            }
        }
        else
        {
            for (tensor_size_t i = 0, indices_size = indices.size(); i < indices_size; ++i)
            {
                subtensor(i) = static_cast<tscalar_return>(this->operator()(indices(i)));
            }
        }
    }

    ///
    /// \brief return a copy of some (sub-)tensors using the given indices.
    /// NB: the indices are relative to the first dimension.
    ///
    template <typename tscalar_return = tscalar>
    auto indexed(indices_cmap_t indices) const
    {
        auto subtensor = tensor_mem_t<tscalar_return, trank>{};
        indexed(indices, subtensor);
        return subtensor;
    } // LCOV_EXCL_LINE

    ///
    /// \brief access an element of the tensor
    ///
    typename tbase::tmutableref operator()(const tensor_size_t index)
    {
        assert(data() != nullptr);
        assert(index >= 0 && index < size());
        return data()[index];
    }

    typename tbase::tconstref operator()(const tensor_size_t index) const
    {
        assert(data() != nullptr);
        assert(index >= 0 && index < size());
        return data()[index];
    }

    template <typename... tindices>
    typename tbase::tmutableref operator()(const tensor_size_t index, const tindices... indices)
    {
        return operator()(offset(index, indices...));
    }

    template <typename... tindices>
    typename tbase::tconstref operator()(const tensor_size_t index, const tindices... indices) const
    {
        return operator()(offset(index, indices...));
    }

    ///
    /// \brief reshape to a new tensor (with the same number of elements).
    /// NB: a single -1 dimension can be inferred from the total size and the remaining positive dimensions!
    ///
    template <typename... tsizes>
    auto reshape(tsizes... sizes)
    {
        return treshape(data(), sizes...);
    }

    template <typename... tsizes>
    auto reshape(tsizes... sizes) const
    {
        return treshape(data(), sizes...);
    }

    ///
    /// \brief iterators for Eigen matrices for STL compatibility.
    ///
    auto begin() { return data(); }

    auto begin() const { return data(); }

    auto end() { return data() + size(); }

    auto end() const { return data() + size(); }

    ///
    /// \brief set all elements to zero.
    ///
    void zero() { array() = tscalar(0); }

    ///
    /// \brief set all elements to the given constant value.
    ///
    auto& full(const tscalar value)
    {
        array() = value;
        return *this;
    }

    ///
    /// \brief set all elements to random values in the [min, max] range.
    ///
    auto& random(const tscalar min = default_min_random<tscalar>(), const tscalar max = +1,
                 const seed_t seed = seed_t{})
    {
        assert(min < max);
        urand(min, max, begin(), end(), seed);
        return *this;
    }

    ///
    /// \brief set all elements in an arithmetic progression from min to max.
    ///
    auto& lin_spaced(const tscalar min, const tscalar max)
    {
        array() = eigen_vector_t<tscalar>::LinSpaced(size(), min, max);
        return *this;
    }

    ///
    /// \brief return true if all values are finite.
    ///
    bool all_finite() const { return vector().allFinite(); }

    ///
    /// \brief return the minimum value.
    ///
    tscalar min() const { return vector().minCoeff(); }

    ///
    /// \brief return the maximum value.
    ///
    tscalar max() const { return vector().maxCoeff(); }

    ///
    /// \brief return the average value.
    ///
    tscalar mean() const { return vector().mean(); }

    ///
    /// \brief return the sum of all its values.
    ///
    tscalar sum() const { return vector().sum(); }

    ///
    /// \brief return the variance of the flatten array.
    ///
    double variance() const
    {
        double variance = 0.0;
        if (size() > 1)
        {
            const auto array   = this->array().template cast<double>();
            const auto count   = static_cast<double>(size());
            const auto average = array.mean();
            variance           = array.square().sum() / count - average * average;
        }
        return variance;
    }

    ///
    /// \brief return the sample standard deviation of the flatten array.
    ///
    double stdev() const
    {
        double stdev = 0.0;
        if (size() > 1)
        {
            const auto count = static_cast<double>(size());
            stdev            = std::sqrt(variance() / (count - 1));
        }
        return stdev;
    }

    ///
    /// \brief return the lp-norm of the flatten array (using the Eigen backend).
    ///
    template <int p>
    auto lpNorm() const
    {
        return vector().template lpNorm<p>();
    }

    ///
    /// \brief return the squared norm of the flatten array (using the Eigen backend).
    ///
    auto squaredNorm() const { return vector().squaredNorm(); }

    ///
    /// \brief return the dot product of the flatten array with the given tensor.
    ///
    auto dot(const tensor_t<tstorage, tscalar, trank>& other) const { return vector().dot(other.vector()); }

    ///
    /// \brief return the dot product of the flatten array with the given Eigen expression.
    ///
    template <typename texpression, std::enable_if_t<is_eigen_v<texpression>, bool> = true>
    auto dot(const texpression& expression) const
    {
        return vector().dot(expression);
    }

    ///
    /// \brief return the Eigen-expression associated to the given row index.
    ///
    auto row(const tensor_size_t row)
    {
        static_assert(trank == 2);
        return matrix().row(row);
    }

    auto row(const tensor_size_t row) const
    {
        static_assert(trank == 2);
        return matrix().row(row);
    }

    ///
    /// \brief return the Eigen-expression associated to the flatten segment [begin, begin + length).
    ///
    auto segment(const tensor_size_t begin, const tensor_size_t length)
    {
        static_assert(trank == 1);
        return vector().segment(begin, length);
    }

    auto segment(const tensor_size_t begin, const tensor_size_t length) const
    {
        static_assert(trank == 1);
        return vector().segment(begin, length);
    }

    ///
    /// \brief return the Eigen-expression associated to the matrix block:
    ///     [row_begin, row_begin + block_rows) x [col_begin, col_begin + block_cols).
    ///
    auto block(const tensor_size_t row_begin, const tensor_size_t col_begin, const tensor_size_t block_rows,
               const tensor_size_t block_cols)
    {
        static_assert(trank == 2);
        return matrix().block(row_begin, col_begin, block_rows, block_cols);
    }

    auto block(const tensor_size_t row_begin, const tensor_size_t col_begin, const tensor_size_t block_rows,
               const tensor_size_t block_cols) const
    {
        static_assert(trank == 2);
        return matrix().block(row_begin, col_begin, block_rows, block_cols);
    }

    ///
    /// \brief return the Eigen-expression associated to the matrix diagonal.
    ///
    auto diagonal()
    {
        static_assert(trank == 2);
        return matrix().diagonal();
    }

    auto diagonal() const
    {
        static_assert(trank == 2);
        return matrix().diagonal();
    }

    ///
    /// \brief return the Eigen-expression associated to the vector or matrix transpose.
    ///
    auto transpose()
    {
        static_assert(trank == 1 || trank == 2);
        if constexpr (trank == 1)
        {
            return vector().transpose();
        }
        else
        {
            return matrix().transpose();
        }
    }

    auto transpose() const
    {
        static_assert(trank == 1 || trank == 2);
        if constexpr (trank == 1)
        {
            return vector().transpose();
        }
        else
        {
            return matrix().transpose();
        }
    }

    ///
    /// \brief multiply element-wise by the given factor.
    ///
    template <typename tscalar_factor, std::enable_if_t<std::is_arithmetic_v<tscalar_factor>, bool> = true>
    tensor_t& operator*=(const tscalar_factor factor)
    {
        vector() *= static_cast<tscalar>(factor);
        return *this;
    }

    ///
    /// \brief divide element-wise by the given factor.
    ///
    template <typename tscalar_factor, std::enable_if_t<std::is_arithmetic_v<tscalar_factor>, bool> = true>
    tensor_t& operator/=(const tscalar_factor factor)
    {
        vector() /= static_cast<tscalar>(factor);
        return *this;
    }

    ///
    /// \brief subtract element-wise the given Eigen expression.
    ///
    template <typename texpression, std::enable_if_t<is_eigen_v<texpression>, bool> = true>
    tensor_t& operator-=(const texpression& expression)
    {
        static_assert(trank == 1 || trank == 2);
        if constexpr (trank == 1)
        {
            vector() -= expression;
        }
        else
        {
            matrix() -= expression;
        }
        return *this;
    }

    ///
    /// \brief subtract element-wise the given tensor.
    ///
    template <template <typename, size_t> class tstorage_rhs>
    tensor_t& operator-=(const tensor_t<tstorage_rhs, tscalar, trank>& rhs)
    {
        assert(dims() == rhs.dims());
        vector() -= rhs.vector();
        return *this;
    }

    ///
    /// \brief add element-wise the given Eigen expression.
    ///
    template <typename texpression, std::enable_if_t<is_eigen_v<texpression>, bool> = true>
    tensor_t& operator+=(const texpression& expression)
    {
        static_assert(trank == 1 || trank == 2);
        if constexpr (trank == 1)
        {
            vector() += expression;
        }
        else
        {
            matrix() += expression;
        }
        return *this;
    }

    ///
    /// \brief add element-wise the given tensor.
    ///
    template <template <typename, size_t> class tstorage_rhs>
    tensor_t& operator+=(const tensor_t<tstorage_rhs, tscalar, trank>& rhs)
    {
        assert(dims() == rhs.dims());
        vector() += rhs.vector();
        return *this;
    }

    ///
    /// \brief construct an Eigen-based vector or matrix expression with all elements zero.
    ///
    static auto zero(const tensor_size_t size)
    {
        static_assert(trank == 1);
        return eigen_vector_t<tscalar>::Zero(size);
    }

    static auto zero(const tensor_size_t rows, const tensor_size_t cols)
    {
        static_assert(trank == 2);
        return eigen_matrix_t<tscalar>::Zero(rows, cols);
    }

    ///
    /// \brief construct an Eigen-based vector or matrix expression with all elements the given constant.
    ///
    template <typename tscalar_value>
    static auto constant(const tensor_size_t size, const tscalar_value value)
    {
        static_assert(trank == 1);
        return eigen_vector_t<tscalar>::Constant(size, static_cast<tscalar>(value));
    }

    template <typename tscalar_value>
    static auto constant(const tensor_size_t rows, const tensor_size_t cols, const tscalar_value value)
    {
        static_assert(trank == 2);
        return eigen_matrix_t<tscalar>::Constant(rows, cols, static_cast<tscalar>(value));
    }

    ///
    /// \brief construct an Eigen-based identity matrix expression.
    ///
    static auto identity(const tensor_size_t rows, const tensor_size_t cols)
    {
        static_assert(trank == 2);
        return eigen_matrix_t<tscalar>::Identity(rows, cols);
    }

private:
    template <typename tdata, typename... tindices>
    auto tvector(tdata ptr, tindices... indices) const
    {
        static_assert(sizeof...(indices) < trank, "invalid number of tensor dimensions");
        return map_vector(ptr + offset0(indices...), ::nano::size(::nano::dims0(dims(), indices...)));
    }

    template <typename tdata, typename... tindices>
    auto tmatrix(tdata ptr, tindices... indices) const
    {
        static_assert(sizeof...(indices) == trank - 2, "invalid number of tensor dimensions");
        return map_matrix(ptr + offset0(indices...), rows(), cols());
    }

    template <typename tdata, typename... tindices>
    auto ttensor(tdata ptr, tindices... indices) const
    {
        static_assert(sizeof...(indices) < trank, "invalid number of tensor dimensions");
        return map_tensor(ptr + offset0(indices...), ::nano::dims0(dims(), indices...));
    }

    template <typename tdata, typename... tsizes>
    auto treshape(tdata ptr, tsizes... sizes) const
    {
        auto dimensions = ::nano::make_dims(sizes...);
        for (auto& dim : dimensions)
        {
            assert(dim == -1 || dim >= 0);
            if (dim == -1)
            {
                dim = -size() / ::nano::size(dimensions);
            }
        }
        assert(::nano::size(dimensions) == size());
        return map_tensor(ptr, dimensions);
    }

    template <typename tdata>
    auto tslice(tdata ptr, tensor_size_t begin, tensor_size_t end) const
    {
        assert(begin >= 0 && begin <= end && end <= this->template size<0>());
        auto dimensions = dims();
        dimensions[0]   = end - begin;
        return map_tensor(ptr + offset0(begin), dimensions);
    }

    template <typename texpression, std::enable_if_t<is_eigen_v<texpression>, bool> = true>
    void assign(const texpression& expression)
    {
        static_assert(trank == 1 || trank == 2);
        if constexpr (trank == 1)
        {
            if constexpr (resizable)
            {
                resize(expression.size());
            }
            assert(size() == expression.size());
            vector() = expression;
        }
        else
        {
            if constexpr (resizable)
            {
                resize(expression.rows(), expression.cols());
            }
            assert(rows() == expression.rows());
            assert(cols() == expression.cols());
            matrix() = expression;
        }
    }
};

///
/// \brief pretty-print the given tensor.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
std::ostream& operator<<(std::ostream& stream, const tensor_t<tstorage, tscalar, trank>& tensor)
{
    return pprint(stream, tensor);
}

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
    std::copy(list.begin(), list.end(), std::begin(tensor));
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
} // LCOV_EXCL_LINE

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
    std::copy(list.begin(), list.end(), std::begin(matrix));
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
    std::copy(list.begin(), list.end(), std::begin(vector));
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
