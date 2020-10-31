#pragma once

#include <nano/tensor/base.h>
#include <nano/tensor/vector.h>

namespace nano
{
    template <typename tscalar>
    class tensor_vector_storage_t;

    template <typename tscalar>
    class tensor_carray_storage_t;

    template <typename tscalar>
    class tensor_marray_storage_t;

    ///
    /// \brief tensor storage using an Eigen vector.
    /// NB: the tensor owns the allocated memory and as such it is resizable.
    ///
    template <typename tscalar, size_t trank>
    class tensor_vector_storage_t : public tensor_base_t<tscalar, trank>
    {
    public:

        tensor_vector_storage_t() = default;
        ~tensor_vector_storage_t() = default;
        tensor_vector_storage_t(const tensor_vector_storage_t&) = default;
        tensor_vector_storage_t(tensor_vector_storage_t&&) noexcept = default;
        tensor_vector_storage_t& operator=(const tensor_vector_storage_t&) = default;
        tensor_vector_storage_t& operator=(tensor_vector_storage_t&&) noexcept = default;

        template <typename... tsizes>
        explicit tensor_vector_storage_t(tsizes... dims) :
            tensor_base_t(dims...),
            m_data(size())
        {
        }

        explicit tensor_vector_storage_t(tdims dims) :
            tensor_base_t(std::move(dims)),
            m_data(size())
        {
        }

        explicit tensor_vector_storage_t(const tensor_carray_storage_t<tscalar, trank>& other) :
            tensor_base_t(other.dims()),
            m_data(map_vector(other.data(), other.size()))
        {
        }

        explicit tensor_vector_storage_t(const tensor_marray_storage_t<tscalar, trank>& other) :
            tensor_base_t(other.dims()),
            m_data(map_vector(other.data(), other.size()))
        {
        }

        tensor_vector_storage_t& operator=(const tensor_carray_storage_t<tscalar, trank>& other)
        {
            assert(size() == other.size());
            if (data() != other.data())
            {
                map_vector(data(), size()) = map_vector(other.data(), other.size());
            }
            return *this;
        }

        tensor_vector_storage_t& operator=(const tensor_marray_storage_t<tscalar, trank>& other)
        {
            assert(size() == other.size());
            if (data() != other.data())
            {
                map_vector(data(), size()) = map_vector(other.data(), other.size());
            }
            return *this;
        }

        template <typename... tsizes>
        void resize(tsizes... dims)
        {
            tensor_base_t::resize(dims...);
            m_data.resize(size());
        }

        void resize(const tdims& dims)
        {
            tensor_base_t::resize(dims);
            m_data.resize(size());
        }

        auto data()
        {
            return m_data.data();
        }
        auto data() const
        {
            return m_data.data();
        }

    private:

        // attributes
        tensor_vector_t<tscalar>    m_data;         ///< store tensor as a 1D vector.
    };

    ///
    /// \brief tensor storage using a constant C-array.
    /// NB: the tensor doesn't own the allocated memory and as such it is not resizable.
    ///
    template <typename tscalar>
    class tensor_carray_storage_t : public tensor_base_t<tscalar>
    {
    public:

        tensor_carray_storage_t() = default;
        ~tensor_carray_storage_t() = default;
        tensor_carray_storage_t(const tensor_carray_storage_t&) = default;
        tensor_carray_storage_t(tensor_carray_storage_t&&) noexcept = default;
        tensor_carray_storage_t& operator=(tensor_carray_storage_t&& other) noexcept = default;

        template <typename... tsizes>
        explicit tensor_carray_storage_t(const tscalar* data, tsizes... dims) :
            tensor_base_t(dims...),
            m_data(data)
        {
        }

        explicit tensor_carray_storage_t(const tscalar* data, tdims dims) :
            tensor_base_t(std::move(dims)),
            m_data(data)
        {
        }

        explicit tensor_carray_storage_t(const tensor_vector_storage_t<tscalar, trank>& other) :
            tensor_base_t(other.dims()),
            m_data(other.data())
        {
        }

        explicit tensor_carray_storage_t(const tensor_marray_storage_t<tscalar, trank>& other) :
            tensor_base_t(other.dims()),
            m_data(other.data())
        {
        }

        tensor_carray_storage_t& operator=(const tensor_vector_storage_t<tscalar, trank>& other) = delete;
        tensor_carray_storage_t& operator=(const tensor_carray_storage_t<tscalar, trank>& other) = delete;
        tensor_carray_storage_t& operator=(const tensor_marray_storage_t<tscalar, trank>& other) = delete;

        template <typename... tsizes>
        void resize(tsizes...) = delete;
        void resize(const tdims&) = delete;

        auto data() const
        {
            return m_data;
        }

    private:

        // attributes
        const tscalar*      m_data{nullptr};    ///< wrap tensor over a contiguous array.
    };

    ///
    /// \brief tensor storage using a mutable C-array.
    /// NB: the tensor doesn't own the allocated memory and as such it is not resizable.
    ///
    template <typename tscalar>
    class tensor_marray_storage_t : public tensor_base_t<tscalar>
    {
    public:

        tensor_marray_storage_t() = default;
        ~tensor_marray_storage_t() = default;
        tensor_marray_storage_t(const tensor_marray_storage_t&) = default;
        tensor_marray_storage_t(tensor_marray_storage_t&&) noexcept = default;
        tensor_marray_storage_t& operator=(tensor_marray_storage_t&& other) noexcept = default;

        template <typename... tsizes>
        explicit tensor_marray_storage_t(tscalar* data, tsizes... dims) :
            tensor_base_t(dims...),
            m_data(data)
        {
        }

        explicit tensor_marray_storage_t(tscalar* data, tdims dims) :
            tensor_base_t(std::move(dims)),
            m_data(data)
        {
        }

        explicit tensor_marray_storage_t(tensor_vector_storage_t<tscalar, trank>& other) :
            tensor_base_t(other.dims()),
            m_data(other.data())
        {
        }

        tensor_marray_storage_t& operator=(const tensor_vector_storage_t<tscalar, trank>& other)
        {
            copy(other);
            return *this;
        }

        tensor_marray_storage_t& operator=(const tensor_carray_storage_t<tscalar, trank>& other)
        {
            copy(other);
            return *this;
        }

        tensor_marray_storage_t& operator=(const tensor_marray_storage_t<tscalar, trank>& other) // NOLINT(cert-oop54-cpp,bugprone-unhandled-self-assignment)
        {
            if (this != &other)
            {
                copy(other);
            }
            return *this;
        }

        template <typename... tsizes>
        void resize(tsizes...) = delete;
        void resize(const tdims&) = delete;

        auto data() const
        {
            return m_data;
        }

    private:

        template <typename tstorage>
        void copy(const tstorage& other)
        {
            assert(size() == other.size());
            if (data() != other.data())
            {
                map_vector(data(), size()) = map_vector(other.data(), other.size());
            }
        }

        // attributes
        tscalar*            m_data{nullptr};    ///< wrap tensor over a contiguous array.
    };
}
