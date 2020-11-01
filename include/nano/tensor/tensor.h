#pragma once

#include <nano/random.h>
#include <nano/tensor/vector.h>
#include <nano/tensor/matrix.h>
#include <nano/tensor/storage.h>

namespace nano
{
    template <typename tstorage, size_t trank>
    class tensor_t;

    ///
    /// \brief tensor that owns the allocated memory.
    ///
    template <typename tscalar, size_t trank>
    using tensor_mem_t = tensor_t<tensor_vector_storage_t<tscalar>, trank>;

    ///
    /// \brief tensor mapping a non-constant array.
    ///
    template <typename tscalar, size_t trank>
    using tensor_map_t = tensor_t<tensor_marray_storage_t<tscalar>, trank>;

    ///
    /// \brief tensor mapping a constant array.
    ///
    template <typename tscalar, size_t trank>
    using tensor_cmap_t = tensor_t<tensor_carray_storage_t<tscalar>, trank>;

    ///
    /// \brief tensor indices.
    ///
    using indices_t = tensor_mem_t<tensor_size_t, 1>;
    using indices_map_t = tensor_map_t<tensor_size_t, 1>;
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
    /// \brief tensor w/o owning the allocated continuous memory.
    ///
    /// NB: all access operations (e.g. Eigen arrays, vectors or matrices or sub-tensors) are performed
    ///     using only continuous memory.
    ///
    template <typename tstorage, size_t trank>
    class tensor_t
    {
    public:

        using tdims = tensor_dims_t<trank>;
        using tscalar = typename tstorage::tscalar;

        using Index = tensor_size_t;    ///< for compatibility with Eigen
        using Scalar = tscalar;         ///< for compatibility with Eigen

        static_assert(trank >= 1, "cannot create tensors with fewer than one dimension");

        ///
        /// \brief default constructor
        ///
        tensor_t()
        {
            m_dims.fill(0);
        }

        ///
        /// \brief constructors that resize the storage to match the given size.
        ///
        template <typename... tsizes>
        explicit tensor_t(tsizes... dims) :
            m_dims({{dims...}}),
            m_storage(this->size())
        {
        }

        explicit tensor_t(const tdims& dims) :
            m_dims(dims),
            m_storage(this->size())
        {
        }

        ///
        /// \brief construct from non-ownning non-const C-array.
        ///
        explicit tensor_t(tscalar* ptr, const tdims& dims) :
            m_dims(dims),
            m_storage(ptr, size())
        {
            assert(ptr != nullptr || !size());
        }

        ///
        /// \brief construct from non-ownning const C-array.
        ///
        explicit tensor_t(const tscalar* ptr, const tdims& dims) :
            m_dims(dims),
            m_storage(ptr, size())
        {
            assert(ptr != nullptr || !size());
        }

        ///
        /// \brief construct from a std::initializer_list.
        ///
        template <typename tscalar_>
        tensor_t(const tdims& dims, const std::initializer_list<tscalar_>& list) :
            m_dims(dims),
            m_storage(this->size())
        {
            assert(this->size() == static_cast<tensor_size_t>(list.size()));
            std::transform(list.begin(), list.end(), data(), [] (auto value) { return static_cast<tscalar>(value); });
        }

        ///
        /// \brief enable copying (delegate to the storage object).
        ///
        tensor_t(const tensor_t&) = default;
        tensor_t& operator=(const tensor_t&) = default;

        ///
        /// \brief enable moving (delegate to the storage object).
        ///
        tensor_t(tensor_t&&) noexcept = default;
        tensor_t& operator=(tensor_t&&) noexcept = default;

        ///
        /// \brief copy constructor from different types (e.g. const array from mutable array)
        ///
        template <typename tstorage2>
        // cppcheck-suppress noExplicitConstructor
        tensor_t(const tensor_t<tstorage2, trank>& other) : // NOLINT(hicpp-explicit-conversions)
            m_dims(other.dims()),
            m_storage(other.data(), other.size())
        {
        }

        ///
        /// \brief copy constructor from different types (e.g. mutable array from mutable vector)
        ///
        template <typename tstorage2>
        // cppcheck-suppress noExplicitConstructor
        tensor_t(tensor_t<tstorage2, trank>& other) : // NOLINT(hicpp-explicit-conversions)
            m_dims(other.dims()),
            m_storage(other.data(), other.size())
        {
        }

        ///
        /// \brief assignment operator from different types (e.g. const from non-const scalars)
        ///
        template <typename tstorage2>
        tensor_t& operator=(const tensor_t<tstorage2, trank>& other)
        {
            m_dims = other.dims();
            m_storage = other.storage();
            return *this;
        }

        ///
        /// \brief default destructor
        ///
        ~tensor_t() = default;

        ///
        /// \brief number of dimensions (aka the rank of the tensor)
        ///
        static constexpr auto rank() { return trank; }

        ///
        /// \brief list of dimensions
        ///
        const auto& dims() const { return m_dims; }

        ///
        /// \brief gather the missing dimensions in a multi-dimensional tensor
        ///     (assuming the last dimensions that are ignored are zero).
        ///
        template <typename... tindices>
        auto dims0(const tindices... indices) const
        {
            static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
            return nano::dims0(dims(), indices...);
        }

        ///
        /// \brief total number of elements.
        ///
        auto size() const { return nano::size(m_dims); }

        ///
        /// \brief number of elements for the given dimension.
        ///
        template <int idim>
        auto size() const { return std::get<idim>(m_dims); }

        ///
        /// \brief interpret the last two dimensions as rows/columns.
        /// NB: e.g. images represented as 3D tensors (color plane, rows, columns)
        /// NB: e.g. ML minibatches represented as 4D tensors (sample, feature plane, rows, columns)
        ///
        auto rows() const { static_assert(trank >= 2 ); return size<trank - 2>(); }
        auto cols() const { static_assert(trank >= 2 ); return size<trank - 1>(); }

        ///
        /// \brief compute the linearized index from the list of offsets.
        ///
        template <typename... tindices>
        auto offset(const tindices... indices) const
        {
            static_assert(sizeof...(indices) == trank, "invalid number of tensor dimensions");
            return nano::index(dims(), indices...);
        }

        ///
        /// \brief compute the linearized index from the list of offsets
        ///     (assuming the last dimensions that are ignored are zero).
        ///
        template <typename... tindices>
        auto offset0(const tindices... indices) const
        {
            static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
            return nano::index0(dims(), indices...);
        }

        ///
        /// \brief resize to new dimensions.
        ///
        template <typename... tsizes>
        tensor_size_t resize(tsizes... dims)
        {
            return resize({{dims...}});
        }

        tensor_size_t resize(const tdims& dims)
        {
            this->m_dims = dims;
            this->m_storage.resize(this->size());
            return this->size();
        }

        ///
        /// \brief set all elements to zero.
        ///
        void zero()
        {
            zero(array());
        }

        ///
        /// \brief set all elements to the given constant value.
        ///
        void constant(const tscalar value)
        {
            constant(array(), value);
        }

        ///
        /// \brief set all elements to random values in the [min, max] range.
        ///
        void random(const tscalar min = -1, const tscalar max = +1)
        {
            random(array(), min, max);
        }

        ///
        /// \brief set all elements in an arithmetic progression from min to max.
        ///
        void lin_spaced(const tscalar min, const tscalar max)
        {
            array() = tensor_vector_t<tscalar>::LinSpaced(size(), min, max);
        }

        ///
        /// \brief returns the minimum value.
        ///
        auto min() const
        {
            return vector().minCoeff();
        }

        ///
        /// \brief returns the maximum value.
        ///
        auto max() const
        {
            return vector().maxCoeff();
        }

        ///
        /// \brief returns the average value.
        ///
        auto mean() const
        {
            return vector().mean();
        }

        ///
        /// \brief returns the sum of all its values.
        ///
        auto sum() const
        {
            return vector().sum();
        }

        ///
        /// \brief returns the variance of all its values.
        ///
        auto variance() const
        {
            double variance = 0.0;
            if (size() > 1)
            {
                const auto array = this->array().template cast<double>();
                const auto count = static_cast<double>(size());
                const auto average = array.mean();
                variance = array.square().sum() / count - average * average;
            }
            return variance;
        }

        ///
        /// \brief returns the sample standard deviation.
        ///
        auto stdev() const
        {
            double stdev = 0.0;
            if (size() > 1)
            {
                const auto count = static_cast<double>(size());
                stdev = std::sqrt(variance() / (count - 1));
            }
            return stdev;
        }

        ///
        /// \brief access the storage container.
        ///
        const auto& storage() const { return m_storage; }

        ///
        /// \brief access the tensor as a continuous C-array
        ///
        auto data() { return m_storage.data(); }
        auto data() const { return m_storage.data(); }

        ///
        /// \brief access the whole tensor as an Eigen vector
        ///
        auto vector() { return map_vector(data(), size()); }
        auto vector() const { return map_vector(data(), size()); }

        ///
        /// \brief access a continuous part of the tensor as an Eigen vector
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto vector(const tindices... indices) { return tvector(data(), indices...); }

        template <typename... tindices>
        auto vector(const tindices... indices) const { return tvector(data(), indices...); }

        ///
        /// \brief access the whole tensor as an Eigen array
        ///
        auto array() { return vector().array(); }
        auto array() const { return vector().array(); }

        ///
        /// \brief access a part of the tensor as an Eigen array
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto array(const tindices... indices) { return vector(indices...).array(); }

        template <typename... tindices>
        auto array(const tindices... indices) const { return vector(indices...).array(); }

        ///
        /// \brief access a part of the tensor as an Eigen matrix
        ///     (assuming that the last two dimensions are ignored)
        ///
        template <typename... tindices>
        auto matrix(const tindices... indices) { return tmatrix(data(), indices...); }

        template <typename... tindices>
        auto matrix(const tindices... indices) const { return tmatrix(data(), indices...); }

        ///
        /// \brief access a part of the tensor as a (sub-)tensor
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto tensor(const tindices... indices) { return ttensor(data(), indices...); }

        template <typename... tindices>
        auto tensor(const tindices... indices) const { return ttensor(data(), indices...); }

        ///
        /// \brief access a part of the tensor as a (sub-)tensor
        ///     (by taking the [begin, end) range of the first dimension)
        ///
        auto slice(tensor_size_t begin, tensor_size_t end)
        {
            return tslice(data(), begin, end);
        }

        auto slice(tensor_size_t begin, tensor_size_t end) const
        {
            return tslice(data(), begin, end);
        }

        auto slice(const tensor_range_t& range)
        {
            return tslice(data(), range.begin(), range.end());
        }

        auto slice(const tensor_range_t& range) const
        {
            return tslice(data(), range.begin(), range.end());
        }

        ///
        /// \brief copy some of (sub-)tensors using the given indices.
        /// NB: the indices are relative to the first dimension.
        ///
        template <typename tindices, typename tscalar_return>
        void indexed(const tindices& indices, tensor_mem_t<tscalar_return, trank>& subtensor) const
        {
            assert(indices.min() >= 0 && indices.max() < size<0>());

            auto dims = this->dims();
            dims[0] = indices.size();

            subtensor.resize(dims);
            if constexpr (trank > 1)
            {
                for (tensor_size_t i = 0, indices_size = indices.size(); i < indices_size; ++ i)
                {
                    subtensor.vector(i) = vector(indices(i)).template cast<tscalar_return>();
                }
            }
            else
            {
                for (tensor_size_t i = 0, indices_size = indices.size(); i < indices_size; ++ i)
                {
                    subtensor(i) = static_cast<tscalar_return>(this->operator()(indices(i)));
                }
            }
        }

        ///
        /// \brief returns a copy of some (sub-)tensors using the given indices.
        /// NB: the indices are relative to the first dimension.
        ///
        template <typename tscalar_return, typename tindices>
        auto indexed(const tindices& indices) const
        {
            auto subtensor = tensor_mem_t<tscalar_return, trank>{};
            indexed(indices, subtensor);
            return subtensor;
        }

        ///
        /// \brief access an element of the tensor
        ///
        tscalar& operator()(const tensor_size_t index)
        {
            assert(const_cast<const tensor_t*>(this)->data() != nullptr);
            assert(index >= 0 && index < size());
            return data()[index];
        }

        tscalar operator()(const tensor_size_t index) const
        {
            assert(data() != nullptr);
            assert(index >= 0 && index < size());
            return data()[index];
        }

        template <typename... tindices>
        tscalar& operator()(const tensor_size_t index, const tindices... indices)
        {
            return operator()(offset(index, indices...));
        }

        template <typename... tindices>
        tscalar operator()(const tensor_size_t index, const tindices... indices) const
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
        /// \brief pretty-print the tensor.
        ///
        std::ostream& print(std::ostream& stream,
            tensor_size_t prefix_space = 0, tensor_size_t prefix_delim = 0, tensor_size_t suffix = 0) const
        {
            [[maybe_unused]] const auto sprint = [&] (const char c, const tensor_size_t count) -> std::ostream&
            {
                for (tensor_size_t i = 0; i < count; ++ i)
                {
                    stream << c;
                }
                return stream;
            };

            if (prefix_space == 0 && prefix_delim == 0 && suffix == 0)
            {
                stream << "shape: " << dims() << "\n";
            }

            if constexpr (trank == 1)
            {
                sprint('[', prefix_delim + 1) << vector().transpose();
                sprint(']', suffix + 1);
            }
            else if constexpr (trank == 2)
            {
                const auto matrix = this->matrix();
                for (tensor_size_t row = 0, rows = matrix.rows(); row < rows; ++ row)
                {
                    if (row == 0)
                    {
                        sprint(' ', prefix_space);
                        sprint('[', prefix_delim + 2);
                    }
                    else
                    {
                        sprint(' ', prefix_space + prefix_delim + 1);
                        sprint('[', 1);
                    }
                    stream << matrix.row(row);

                    if (row + 1 < rows)
                    {
                        stream << "]\n";
                    }
                    else
                    {
                        sprint(']', suffix + 2);
                    }
                }
            }
            else
            {
                for (tensor_size_t row = 0, rows = size<0>(); row < rows; ++ row)
                {
                    tensor(row).print(stream,
                        (row == 0) ? prefix_space : (prefix_space + prefix_delim + 1),
                        (row == 0) ? (prefix_delim + 1) : 0,
                        suffix);

                    if (row + 1 < rows)
                    {
                        stream << "\n";
                    }
                    else
                    {
                        stream << "]";
                    }
                }
            }
            return stream;
        }

    private:

        template <typename tarray>
        static void zero(tarray&& array)
        {
            array.setZero();
        }

        template <typename tarray, typename tscalar>
        static void constant(tarray&& array, const tscalar value)
        {
            array.setConstant(value);
        }

        template <typename tarray, typename tscalar>
        static void random(tarray&& array, const tscalar min, const tscalar max)
        {
            assert(min < max);
            urand(min, max, array.data(), array.data() + array.size(), make_rng());
        }

        template <typename tdata, typename... tindices>
        auto tvector(tdata ptr, const tindices... indices) const
        {
            static_assert(sizeof...(indices) < trank, "invalid number of tensor dimensions");
            return map_vector(ptr + offset0(indices...), nano::size(nano::dims0(dims(), indices...)));
        }

        template <typename tdata, typename... tindices>
        auto tmatrix(tdata ptr, const tindices... indices) const
        {
            static_assert(sizeof...(indices) == trank - 2, "invalid number of tensor dimensions");
            return map_matrix(ptr + offset0(indices...), rows(), cols());
        }

        template <typename tdata, typename... tindices>
        auto ttensor(tdata ptr, const tindices... indices) const
        {
            static_assert(sizeof...(indices) < trank, "invalid number of tensor dimensions");
            return map_tensor(ptr + offset0(indices...), nano::dims0(dims(), indices...));
        }

        template <typename tdata, typename... tsizes>
        auto treshape(tdata ptr, tsizes... sizes) const
        {
            auto dims = nano::make_dims(sizes...);
            for (auto& dim : dims)
            {
                assert(dim == -1 || dim >= 0);
                if (dim == -1)
                {
                    dim = -size() / nano::size(dims);
                }
            }
            assert(nano::size(dims) == size());
            return map_tensor(ptr, dims);
        }

        template <typename tdata>
        auto tslice(tdata ptr, const tensor_size_t begin, const tensor_size_t end) const
        {
            assert(begin >= 0 && begin < end && end <= size<0>());
            auto dims = this->dims();
            dims[0] = end - begin;
            return map_tensor(ptr + offset0(begin), dims);
        }

        // attributes
        tdims           m_dims{};   ///<
        tstorage        m_storage;  ///<
    };

    ///
    /// \brief iterators for STL compatibility.
    ///
    template <typename tstorage, size_t trank>
    auto begin(tensor_t<tstorage, trank>& m)
    {
        return m.data();
    }

    template <typename tstorage, size_t trank>
    auto begin(const tensor_t<tstorage, trank>& m)
    {
        return m.data();
    }

    template <typename tstorage, size_t trank>
    auto end(tensor_t<tstorage, trank>& m)
    {
        return m.data() + m.size();
    }

    template <typename tstorage, size_t trank>
    auto end(const tensor_t<tstorage, trank>& m)
    {
        return m.data() + m.size();
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
    /// \brief compare two tensors element-wise.
    ///
    template <typename tstorage1, typename tstorage2, size_t trank>
    bool operator==(const tensor_t<tstorage1, trank>& lhs, const tensor_t<tstorage2, trank>& rhs)
    {
        return  lhs.dims() == rhs.dims() &&
                lhs.vector() == rhs.vector();
    }

    ///
    /// \brief pretty-print the given tensor.
    ///
    template <typename tstorage, size_t trank>
    std::ostream& operator<<(std::ostream& stream, const tensor_t<tstorage, trank>& tensor)
    {
        return tensor.print(stream);
    }
}
