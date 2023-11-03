#pragma once

#include <nano/core/numeric.h>
#include <nano/core/random.h>
#include <nano/tensor/eigen.h>
#include <nano/tensor/range.h>
#include <nano/tensor/storage.h>

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
    tensor_t(const texpression& expression)
    {
        static_assert(trank == 1 || trank == 2);
        if constexpr (trank == 1)
        {
            resize(expression.size());
            vector() = expression;
        }
        else
        {
            resize(expression.rows(), expression.cols());
            matrix() = expression;
        }
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
        static_assert(trank == 1 || trank == 2);
        if constexpr (trank == 1)
        {
            vector() = expression;
        }
        else
        {
            matrix() = expression;
        }
        return *this;
    }

    ///
    /// \brief default destructor
    ///
    ~tensor_t() = default;

    ///
    /// \brief resize to new dimensions.
    ///
    template <typename... tsizes>
    void resize(tsizes... dimensions)
    {
        tbase::resize(make_dims(dimensions...));
    }

    void resize(const tdims& dimensions) { tbase::resize(dimensions); }

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
    /// \brief returns a copy of some (sub-)tensors using the given indices.
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
        random(array(), min, max, seed);
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
    /// \brief returns the minimum value.
    ///
    auto min() const { return vector().minCoeff(); }

    ///
    /// \brief returns the maximum value.
    ///
    auto max() const { return vector().maxCoeff(); }

    ///
    /// \brief returns the average value.
    ///
    auto mean() const { return vector().mean(); }

    ///
    /// \brief returns the sum of all its values.
    ///
    auto sum() const { return vector().sum(); }

    ///
    /// \brief returns the variance of the flatten array.
    ///
    auto variance() const
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
    /// \brief returns the sample standard deviation of the flatten array.
    ///
    auto stdev() const
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
    /// \brief returns the lp-norm of the flatten array (using the Eigen backend).
    ///
    template <int p>
    auto lpNorm() const
    {
        return vector().template lpNorm<p>();
    }

    ///
    /// \brief returns the squared norm of the flatten array (using the Eigen backend).
    ///
    auto squaredNorm() const { return vector().squaredNorm(); }

    ///
    /// \brief returns the dot product of the flatten array with the given tensor.
    ///
    auto dot(const tensor_t<tstorage, tscalar, trank>& other) const { return vector().dot(other.vector()); }

    ///
    /// \brief returns the dot product of the flatten array with the given Eigen expression.
    ///
    template <typename texpression, std::enable_if_t<is_eigen_v<texpression>, bool> = true>
    auto dot(const texpression& expression) const
    {
        return vector().dot(expression);
    }

    ///
    /// \brief returns the Eigen-expression associated to the flatten segment [begin, end).
    ///
    auto segment(const tensor_size_t begin, const tensor_size_t end) { return vector().segment(begin, end); }

    auto segment(const tensor_size_t begin, const tensor_size_t end) const { return vector().segment(begin, end); }

    ///
    /// \brief returns the Eigen-expression associated to the matrix block:
    ///     [row_begin, row_begin + block_rows) x [col_begin, col_begin + block_cols).
    ///
    auto block(const tensor_size_t row_begin, const tensor_size_t col_begin, const tensor_size_t block_rows,
               const tensor_size_t block_cols)
    {
        return matrix().block(row_begin, col_begin, block_rows, block_cols);
    }

    auto block(const tensor_size_t row_begin, const tensor_size_t col_begin, const tensor_size_t block_rows,
               const tensor_size_t block_cols) const
    {
        return matrix().block(row_begin, col_begin, block_rows, block_cols);
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
    template <typename tarray>
    static void random(tarray&& array, tscalar min, tscalar max, seed_t seed)
    {
        assert(min < max);
        urand(min, max, array.data(), array.data() + array.size(), make_rng(seed));
    }

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
        assert(begin >= 0 && begin < end && end <= this->template size<0>());
        auto dimensions = dims();
        dimensions[0]   = end - begin;
        return map_tensor(ptr + offset0(begin), dimensions);
    }
};

///
/// \brief iterators for STL compatibility.
///
template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
auto begin(tensor_t<tstorage, tscalar, trank>& m)
{
    return m.data();
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
auto begin(const tensor_t<tstorage, tscalar, trank>& m)
{
    return m.data();
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
auto end(tensor_t<tstorage, tscalar, trank>& m)
{
    return m.data() + m.size();
}

template <template <typename, size_t> class tstorage, typename tscalar, size_t trank>
auto end(const tensor_t<tstorage, tscalar, trank>& m)
{
    return m.data() + m.size();
}

///
/// \brief returns true if the two tensors are close, ignoring not-finite values if present.
///
template <template <typename, size_t> class tstorage1, template <typename, size_t> class tstorage2, typename tscalar,
          size_t trank>
bool close(const tensor_t<tstorage1, tscalar, trank>& lhs, const tensor_t<tstorage2, tscalar, trank>& rhs,
           const double epsilon)
{
    if (lhs.dims() != rhs.dims())
    {
        return false;
    }
    for (tensor_size_t i = 0, size = lhs.size(); i < size; ++i)
    {
        const auto lhs_finite = ::nano::isfinite(lhs(i));
        const auto rhs_finite = ::nano::isfinite(rhs(i));
        if ((lhs_finite != rhs_finite) ||
            (lhs_finite && !close(static_cast<double>(lhs(i)), static_cast<double>(rhs(i)), epsilon)))
        {
            return false;
        }
    }
    return true;
}
} // namespace nano
