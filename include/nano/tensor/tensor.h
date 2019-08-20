#pragma once

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
    /// \brief map non-constant data to tensors
    ///
    template <typename tscalar_, size_t trank>
    auto map_tensor(tscalar_* data, const tensor_dims_t<trank>& dims)
    {
        using tscalar = typename std::remove_const<tscalar_>::type;
        return tensor_map_t<tscalar, trank>(data, dims);
    }

    ///
    /// \brief map constant data to tensors
    ///
    template <typename tscalar_, size_t trank>
    auto map_tensor(const tscalar_* data, const tensor_dims_t<trank>& dims)
    {
        using tscalar = typename std::remove_const<tscalar_>::type;
        return tensor_cmap_t<const tscalar, trank>(data, dims);
    }

    ///
    /// \brief map non-constant data to tensors
    ///
    template <typename tscalar_, typename... tsizes>
    auto map_tensor(tscalar_* data, const tsizes... dims)
    {
        return map_tensor(data, make_dims(dims...));
    }

    ///
    /// \brief map constant data to tensors
    ///
    template <typename tscalar_, typename... tsizes>
    auto map_tensor(const tscalar_* data, const tsizes... dims)
    {
        return map_tensor(data, make_dims(dims...));
    }

    ///
    /// \brief tensor.
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
        /// \brief constructors that resize the storage to match the given size
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
        /// \brief constructors that maps const or non-const arrays
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
        tensor_t(const tensor_t<tstorage2, trank>& other) :
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
        /// \brief total number of elements
        ///
        auto size() const { return nano::size(m_dims); }

        ///
        /// \brief number of elements for the given dimension
        ///
        template <int idim>
        auto size() const { return std::get<idim>(m_dims); }

        ///
        /// \brief interpret the last two dimensions as rows/columns
        /// NB: e.g. images represented as 3D tensors (color plane, rows, columns)
        /// NB: e.g. ML minibatches represented as 4D tensors (sample, feature plane, rows, columns)
        ///
        auto rows() const { static_assert(trank >= 2, ""); return size<trank - 2>(); }
        auto cols() const { static_assert(trank >= 2, ""); return size<trank - 1>(); }

        ///
        /// \brief compute the linearized index from the list of offsets
        ///
        template <typename... tindices>
        auto offset(const tindices... indices) const
        {
            static_assert(sizeof...(indices) == trank, "invalid number of tensor dimensions");
            return nano::index(dims(), indices...);
        }

        ///
        /// \brief compute the linearized index from the list of offsets
        ///     (assuming the last dimensions that are ignored are zero)
        ///
        template <typename... tindices>
        auto offset0(const tindices... indices) const
        {
            static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
            return nano::index0(dims(), indices...);
        }

        ///
        /// \brief resize to new dimensions
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
        /// \brief set all elements to zero
        ///
        void zero() { zero(array()); }
        void setZero() { zero(); }

        ///
        /// \brief set all elements to a constant value
        ///
        void constant(const tscalar value) { constant(array(), value); }
        void setConstant(const tscalar value) { constant(value); }

        ///
        /// \brief set all elements to random values in the [min, max] range
        ///
        void random(const tscalar min = -1, const tscalar max = +1) { random(array(), min, max); }
        void setRandom(const tscalar min = -1, const tscalar max = +1) { random(min, max); }

        ///
        /// \brief access the storage container
        ///
        const auto& storage() const { return m_storage; }

        ///
        /// \brief access the tensor as a C-array
        ///
        auto data() const { return m_storage.data(); }
        auto data() { return m_storage.data(); }

        ///
        /// \brief access the tensor as a vector
        ///     (assuming the last dimensions that are ignored are zero).
        ///
        auto vector() const { return map_vector(data(), size()); }
        auto vector() { return map_vector(data(), size()); }

        template <typename... tindices>
        auto vector(const tindices... indices) const { return tvector(data(), indices...); }

        template <typename... tindices>
        auto vector(const tindices... indices) { return tvector(data(), indices...); }

        ///
        /// \brief access the tensor as an array
        ///     (assuming the last dimensions that are ignored are zero).
        ///
        auto array() const { return vector().array(); }
        auto array() { return vector().array(); }

        template <typename... tindices>
        auto array(const tindices... indices) const { return vector(indices...).array(); }

        template <typename... tindices>
        auto array(const tindices... indices) { return vector(indices...).array(); }

        ///
        /// \brief access the tensor as a matrix
        ///     (assuming that the last two dimensions are ignored).
        ///
        template <typename... tindices>
        auto matrix(const tindices... indices) const { return tmatrix(data(), indices...); }

        template <typename... tindices>
        auto matrix(const tindices... indices) { return tmatrix(data(), indices...); }

        ///
        /// \brief access the tensor as a (sub-)tensor
        ///     (assuming the last dimensions that are ignored are zero).
        ///
        template <typename... tindices>
        auto tensor(const tindices... indices) const { return ttensor(data(), indices...); }

        template <typename... tindices>
        auto tensor(const tindices... indices) { return ttensor(data(), indices...); }

        ///
        /// \brief returns a copy of some (sub-)tensors using the given indices.
        ///
        /// NB: the indices are relative to the first dimension.
        ///
        template <typename tindices>
        auto indexed(const tindices& indices) const
        {
            assert(indices.minCoeff() >= 0 && indices.maxCoeff() < size<0>());

            auto dims = this->dims();
            dims[0] = indices.size();

            auto subtensor = tensor_mem_t<tscalar, trank>{dims};
            for (tensor_size_t i = 0, indices_size = indices.size(); i < indices_size; ++ i)
            {
                subtensor.tensor(i) = tensor(indices(i));
            }

            return subtensor;
        }

        ///
        /// \brief access an element of the tensor
        ///
        tconst_reference operator()(const tensor_size_t index) const
        {
            assert(data() != nullptr);
            assert(index >= 0 && index < size());
            return data()[index];
        }

        treference operator()(const tensor_size_t index)
        {
            assert(const_cast<const tstorage&>(m_storage).data() != nullptr);
            assert(index >= 0 && index < size());
            return data()[index];
        }

        template <typename... tindices>
        tconst_reference operator()(const tensor_size_t index, const tindices... indices) const
        {
            return operator()(offset(index, indices...));
        }

        template <typename... tindices>
        treference operator()(const tensor_size_t index, const tindices... indices)
        {
            return operator()(offset(index, indices...));
        }

        ///
        /// \brief reshape to a new tensor (with the same number of elements)
        ///
        template <typename... tsizes>
        auto reshape(const tsizes... sizes) const { return treshape(data(), sizes...); }

        template <typename... tsizes>
        auto reshape(const tsizes... sizes) { return treshape(data(), sizes...); }

        ///
        /// \brief iterators for Eigen matrices for STL compatibility.
        ///
        auto begin() { return data(); }
        auto begin() const { return data(); }

        auto end() { return data() + size(); }
        auto end() const { return data() + size(); }

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
            array.setRandom(); // [-1, +1]
            array = (array + 1) * (max - min) / 2 + min;
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
            assert(nano::size(nano::make_dims(sizes...)) == size());
            return map_tensor(ptr, sizes...);
        }

        // attributes
        tdims           m_dims;
        tstorage        m_storage;
    };
}
