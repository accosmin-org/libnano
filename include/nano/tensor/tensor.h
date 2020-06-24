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
    using tensor_mem_t = tensor_t<tensor_vstorage_t<tscalar>, trank>;

    ///
    /// \brief tensor mapping a non-constant array.
    ///
    template <typename tscalar, size_t trank>
    using tensor_map_t = tensor_t<tensor_pstorage_t<tscalar>, trank>;

    ///
    /// \brief tensor mapping a constant array.
    ///
    template <typename tscalar, size_t trank>
    using tensor_cmap_t = tensor_t<tensor_pstorage_t<const tscalar>, trank>;

    ///
    /// \brief tensor indices.
    ///
    using indices_t = tensor_mem_t<tensor_size_t, 1>;

    ///
    /// \brief map non-constant data to tensors.
    ///
    template <typename tscalar_, size_t trank>
    auto map_tensor(tscalar_* data, const tensor_dims_t<trank>& dims)
    {
        using tscalar = typename std::remove_const<tscalar_>::type;
        return tensor_map_t<tscalar, trank>(data, dims);
    }

    ///
    /// \brief map constant data to tensors.
    ///
    template <typename tscalar_, size_t trank>
    auto map_tensor(const tscalar_* data, const tensor_dims_t<trank>& dims)
    {
        using tscalar = typename std::remove_const<tscalar_>::type;
        return tensor_cmap_t<const tscalar, trank>(data, dims);
    }

    ///
    /// \brief map non-constant data to tensors.
    ///
    template <typename tscalar, typename... tsizes>
    auto map_tensor(tscalar* data, const tsizes... dims) // NOLINT(readability-avoid-const-params-in-decls)
    {
        return map_tensor(data, make_dims(dims...));
    }

    ///
    /// \brief map constant data to tensors.
    ///
    template <typename tscalar, typename... tsizes>
    auto map_tensor(const tscalar* data, const tsizes... dims) // NOLINT(readability-avoid-const-params-in-decls)
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
        using treference = typename tstorage::treference;
        using tconst_reference = typename tstorage::tconst_reference;

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
        explicit tensor_t(const tsizes... dims) :
            m_dims({{dims...}}),
            m_storage(this->size())
        {
            static_assert(tstorage::resizable, "tensor not resizable");
        }

        explicit tensor_t(const tdims& dims) :
            m_dims(dims),
            m_storage(this->size())
        {
            static_assert(tstorage::resizable, "tensor not resizable");
        }

        ///
        /// \brief constructors that maps const or non-const arrays.
        ///
        explicit tensor_t(tscalar* ptr, const tdims& dims) :
            m_dims(dims),
            m_storage(ptr, size())
        {
            static_assert(!tstorage::resizable, "tensor resizable");
            assert(ptr != nullptr || !size());
        }

        explicit tensor_t(const tscalar* ptr, const tdims& dims) :
            m_dims(dims),
            m_storage(ptr, size())
        {
            static_assert(!tstorage::resizable, "tensor resizable");
            assert(ptr != nullptr || !size());
        }

        ///
        /// \brief construct from a std::array (only for 1D tensors).
        ///
        template <typename tscalar_, size_t N>
        // cppcheck-suppress noExplicitConstructor
        tensor_t(const std::array<tscalar_, N>& array) : // NOLINT(hicpp-explicit-conversions)
            m_storage(static_cast<tensor_size_t>(N))
        {
            static_assert(tstorage::resizable, "tensor resizable");
            static_assert(trank == 1, "tensor must be of rank 1 to be initialized from std::array");
            m_dims[0] = static_cast<tensor_size_t>(N);
            vector() = map_vector(array.data(), this->size()).template cast<tscalar>();
        }

        ///
        /// \brief construct from a std::array.
        ///
        template <typename tscalar_, size_t N>
        tensor_t(const tdims& dims, const std::array<tscalar_, N>& array) :
            m_dims(dims),
            m_storage(this->size())
        {
            static_assert(tstorage::resizable, "tensor resizable");
            assert(this->size() == static_cast<tensor_size_t>(N));
            vector() = map_vector(array.data(), this->size()).template cast<tscalar>();
        }

        ///
        /// \brief default for copying and moving (delegate to the storage objects)
        ///
        tensor_t(const tensor_t&) = default;
        tensor_t& operator=(const tensor_t&) = default;

        tensor_t(tensor_t&&) noexcept = default;
        tensor_t& operator=(tensor_t&&) noexcept = default;

        ///
        /// \brief copy constructor from different types (e.g. const from non-const scalars)
        ///
        template <typename tstorage2>
        // cppcheck-suppress noExplicitConstructor
        tensor_t(const tensor_t<tstorage2, trank>& other) : // NOLINT(hicpp-explicit-conversions)
            m_dims(other.dims()),
            m_storage(other.storage())
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
        [[nodiscard]] const auto& dims() const { return m_dims; }

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
        [[nodiscard]] auto size() const { return nano::size(m_dims); }

        ///
        /// \brief number of elements for the given dimension.
        ///
        template <int idim>
        [[nodiscard]] auto size() const { return std::get<idim>(m_dims); }

        ///
        /// \brief interpret the last two dimensions as rows/columns.
        /// NB: e.g. images represented as 3D tensors (color plane, rows, columns)
        /// NB: e.g. ML minibatches represented as 4D tensors (sample, feature plane, rows, columns)
        ///
        [[nodiscard]] auto rows() const { static_assert(trank >= 2 ); return size<trank - 2>(); }
        [[nodiscard]] auto cols() const { static_assert(trank >= 2 ); return size<trank - 1>(); }

        ///
        /// \brief compute the linearized index from the list of offsets.
        ///
        template <typename... tindices>
        [[nodiscard]] auto offset(const tindices... indices) const
        {
            static_assert(sizeof...(indices) == trank, "invalid number of tensor dimensions");
            return nano::index(dims(), indices...);
        }

        ///
        /// \brief compute the linearized index from the list of offsets
        ///     (assuming the last dimensions that are ignored are zero).
        ///
        template <typename... tindices>
        [[nodiscard]] auto offset0(const tindices... indices) const
        {
            static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
            return nano::index0(dims(), indices...);
        }

        ///
        /// \brief resize to new dimensions.
        ///
        template <typename... tsizes>
        tensor_size_t resize(const tsizes... dims)
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
        /// \brief set all elements to a constant value.
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
        [[nodiscard]] auto min() const
        {
            return vector().minCoeff();
        }

        ///
        /// \brief returns the maximum value.
        ///
        [[nodiscard]] auto max() const
        {
            return vector().maxCoeff();
        }

        ///
        /// \brief access the storage container.
        ///
        [[nodiscard]] const auto& storage() const { return m_storage; }

        ///
        /// \brief access the tensor as a continuous C-array
        ///
        auto data() { return m_storage.data(); }
        [[nodiscard]] auto data() const { return m_storage.data(); }

        ///
        /// \brief access the whole tensor as an Eigen vector
        ///
        auto vector() { return map_vector(data(), size()); }
        [[nodiscard]] auto vector() const { return map_vector(data(), size()); }

        ///
        /// \brief access a continuous part of the tensor as an Eigen vector
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto vector(const tindices... indices) { return tvector(data(), indices...); }

        template <typename... tindices>
        [[nodiscard]] auto vector(const tindices... indices) const { return tvector(data(), indices...); }

        ///
        /// \brief access the whole tensor as an Eigen array
        ///
        auto array() { return vector().array(); }
        [[nodiscard]] auto array() const { return vector().array(); }

        ///
        /// \brief access a part of the tensor as an Eigen array
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto array(const tindices... indices) { return vector(indices...).array(); }

        template <typename... tindices>
        [[nodiscard]] auto array(const tindices... indices) const { return vector(indices...).array(); }

        ///
        /// \brief access a part of the tensor as an Eigen matrix
        ///     (assuming that the last two dimensions are ignored)
        ///
        template <typename... tindices>
        auto matrix(const tindices... indices) { return tmatrix(data(), indices...); }

        template <typename... tindices>
        [[nodiscard]] auto matrix(const tindices... indices) const { return tmatrix(data(), indices...); }

        ///
        /// \brief access a part of the tensor as a (sub-)tensor
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto tensor(const tindices... indices) { return ttensor(data(), indices...); }

        template <typename... tindices>
        [[nodiscard]] auto tensor(const tindices... indices) const { return ttensor(data(), indices...); }

        ///
        /// \brief access a part of the tensor as a (sub-)tensor
        ///     (by taking the [begin, begin + delta) range of the first dimension)
        ///
        [[nodiscard]] auto slice(const tensor_size_t begin, const tensor_size_t delta)
        {
            return tslice(data(), begin, begin + delta);
        }

        [[nodiscard]] auto slice(const tensor_size_t begin, const tensor_size_t delta) const
        {
            return tslice(data(), begin, begin + delta);
        }

        [[nodiscard]] auto slice(const tensor_range_t& range)
        {
            return slice(range.begin(), range.size());
        }

        [[nodiscard]] auto slice(const tensor_range_t& range) const
        {
            return slice(range.begin(), range.size());
        }

        ///
        /// \brief copy some of (sub-)tensors using the given indices.
        /// NB: the indices are relative to the first dimension.
        ///
        template <typename tscalar_return>
        void indexed(const indices_t& indices, tensor_mem_t<tscalar_return, trank>& subtensor) const
        {
            assert(indices.min() >= 0 && indices.max() < size<0>());

            auto dims = this->dims();
            dims[0] = indices.size();

            subtensor.resize(dims);
            for (tensor_size_t i = 0, indices_size = indices.size(); i < indices_size; ++ i)
            {
                subtensor.vector(i) = vector(indices(i)).template cast<tscalar_return>();
            }
        }

        ///
        /// \brief returns a copy of some (sub-)tensors using the given indices.
        /// NB: the indices are relative to the first dimension.
        ///
        template <typename tscalar_return, typename tindices>
        [[nodiscard]] auto indexed(const tindices& indices) const
        {
            auto subtensor = tensor_mem_t<tscalar_return, trank>{};
            indexed(indices, subtensor);
            return subtensor;
        }

        ///
        /// \brief access an element of the tensor
        ///
        treference operator()(const tensor_size_t index)
        {
            assert(const_cast<const tensor_t*>(this)->data() != nullptr);
            assert(index >= 0 && index < size());
            return data()[index];
        }

        tconst_reference operator()(const tensor_size_t index) const
        {
            assert(data() != nullptr);
            assert(index >= 0 && index < size());
            return data()[index];
        }

        template <typename... tindices>
        treference operator()(const tensor_size_t index, const tindices... indices)
        {
            return operator()(offset(index, indices...));
        }


        template <typename... tindices>
        tconst_reference operator()(const tensor_size_t index, const tindices... indices) const
        {
            return operator()(offset(index, indices...));
        }
        ///
        /// \brief reshape to a new tensor (with the same number of elements).
        /// NB: a single -1 dimension can be inferred from the total size and the remaining positive dimensions!
        ///
        template <typename... tsizes>
        auto reshape(const tsizes... sizes)
        {
            return treshape(data(), sizes...);
        }

        template <typename... tsizes>
        [[nodiscard]] auto reshape(const tsizes... sizes) const
        {
            return treshape(data(), sizes...);
        }

        ///
        /// \brief iterators for Eigen matrices for STL compatibility.
        ///
        auto begin() { return data(); }
        [[nodiscard]] auto begin() const { return data(); }

        auto end() { return data() + size(); }
        [[nodiscard]] auto end() const { return data() + size(); }

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
        auto treshape(tdata ptr, const tsizes... sizes) const
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
    template <typename tstorage, size_t trank>
    bool operator==(const tensor_t<tstorage, trank>& lhs, const tensor_t<tstorage, trank>& rhs)
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
