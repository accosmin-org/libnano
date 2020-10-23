#pragma once

#include <nano/tensor/index.h>
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
    /// \brief tensor storage base class.
    ///
    template
    <
        typename tscalar,
        typename tscalar_remove_cvref = typename std::remove_cv<typename std::remove_reference<tscalar>::type>::type,
        typename = typename std::enable_if<std::is_same<tscalar, tscalar_remove_cvref>::value>::type
    >
    class tensor_storage_t
    {
    public:

        using treference = tscalar&;
        using tconst_reference = const tscalar&;
    };

    ///
    /// \brief tensor storage using an Eigen vector.
    /// NB: the tensor owns the allocated memory and as such the tensor is resizable.
    ///
    template <typename tscalar>
    class tensor_vector_storage_t : public tensor_storage_t<tscalar>
    {
    public:

        tensor_vector_storage_t() = default;
        ~tensor_vector_storage_t() = default;
        tensor_vector_storage_t(const tensor_vector_storage_t&) = default;
        tensor_vector_storage_t(tensor_vector_storage_t&&) noexcept = default;
        tensor_vector_storage_t& operator=(const tensor_vector_storage_t&) = default;
        tensor_vector_storage_t& operator=(tensor_vector_storage_t&&) noexcept = default;

        explicit tensor_vector_storage_t(tensor_size_t size) :
            m_data(size)
        {
        }

        explicit tensor_vector_storage_t(const tensor_vector_t<tscalar>& data) :
            m_data(data)
        {
        }

        explicit tensor_vector_storage_t(const tensor_carray_storage_t<tscalar>& other) :
            m_data(map_vector(other.data(), other.size()))
        {
        }

        explicit tensor_vector_storage_t(const tensor_marray_storage_t<tscalar>& other) :
            m_data(map_vector(other.data(), other.size()))
        {
        }

        tensor_vector_storage_t& operator=(const tensor_carray_storage_t<tscalar>& other)
        {
            if (m_data.data() != other.data())
            {
                m_data = map_vector(other.data(), other.size());
            }
            return *this;
        }

        tensor_vector_storage_t& operator=(const tensor_marray_storage_t<tscalar>& other)
        {
            if (m_data.data() != other.data())
            {
                m_data = map_vector(other.data(), other.size());
            }
            return *this;
        }

        auto data() { return m_data.data(); }
        auto data() const { return m_data.data(); }
        auto size() const { return m_data.size(); }
        void resize(tensor_size_t size) { m_data.resize(size); }

    private:

        // attributes
        tensor_vector_t<tscalar>    m_data;         ///< store tensor as a 1D vector.
    };

    ///
    /// \brief tensor storage using a constant C-array.
    /// NB: the tensor doesn't own the allocated memory and as such is not resizable.
    ///
    template <typename tscalar>
    class tensor_carray_storage_t : public tensor_storage_t<tscalar>
    {
    public:

        tensor_carray_storage_t() = default;
        ~tensor_carray_storage_t() = default;
        tensor_carray_storage_t(const tensor_carray_storage_t&) = default;
        tensor_carray_storage_t(tensor_carray_storage_t&&) noexcept = default;
        tensor_carray_storage_t& operator=(tensor_carray_storage_t&& other) noexcept = default;

        tensor_carray_storage_t(const tscalar* data, tensor_size_t size) :
            m_data(data), m_size(size)
        {
        }

        explicit tensor_carray_storage_t(const tensor_vector_storage_t<tscalar>& other) :
            m_data(other.data()), m_size(other.size())
        {
        }

        explicit tensor_carray_storage_t(const tensor_marray_storage_t<tscalar>& other) :
            m_data(other.data()), m_size(other.size())
        {
        }

        tensor_carray_storage_t& operator=(const tensor_vector_storage_t<tscalar>& other) = delete;
        tensor_carray_storage_t& operator=(const tensor_carray_storage_t<tscalar>& other) = delete;
        tensor_carray_storage_t& operator=(const tensor_marray_storage_t<tscalar>& other) = delete;

        auto data() const { return m_data; }
        auto size() const { return m_size; }
        void resize(tensor_size_t) = delete;

    private:

        // attributes
        const tscalar*      m_data{nullptr};    ///< wrap tensor over a contiguous array.
        tensor_size_t       m_size{0};          ///<
    };

    ///
    /// \brief tensor storage using a mutable C-array.
    /// NB: the tensor doesn't own the allocated memory and as such is not resizable.
    ///
    template <typename tscalar>
    class tensor_marray_storage_t : public tensor_storage_t<tscalar>
    {
    public:

        tensor_marray_storage_t() = default;
        ~tensor_marray_storage_t() = default;
        tensor_marray_storage_t(const tensor_marray_storage_t&) = default;
        tensor_marray_storage_t(tensor_marray_storage_t&&) noexcept = default;
        tensor_marray_storage_t& operator=(tensor_marray_storage_t&& other) noexcept = default;

        tensor_marray_storage_t(tscalar* data, tensor_size_t size) :
            m_data(data), m_size(size)
        {
        }

        explicit tensor_marray_storage_t(tensor_vector_storage_t<tscalar>& other) :
            m_data(other.data()), m_size(other.size())
        {
        }

        tensor_marray_storage_t& operator=(const tensor_vector_storage_t<tscalar>& other)
        {
            return copy(other);
        }

        tensor_marray_storage_t& operator=(const tensor_carray_storage_t<tscalar>& other)
        {
            return copy(other);
        }

        tensor_marray_storage_t& operator=(const tensor_marray_storage_t<tscalar>& other)
        {
            return copy(other);
        }

        auto data() const { return m_data; }
        auto size() const { return m_size; }
        void resize(tensor_size_t) = delete;

    private:

        template <typename tstorage>
        tensor_marray_storage_t& copy(const tstorage& other)
        {
            assert(size() == other.size());
            if (data() != other.data())
            {
                map_vector(data(), size()) = map_vector(other.data(), other.size());
            }
            return *this;
        }

        // attributes
        tscalar*            m_data{nullptr};    ///< wrap tensor over a contiguous array.
        tensor_size_t       m_size{0};          ///<
    };
}
