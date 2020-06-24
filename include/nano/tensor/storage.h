#pragma once

#include <nano/tensor/index.h>

namespace nano
{
    template <typename tscalar_>
    class tensor_pstorage_t;

    ///
    /// \brief tensor storage using an Eigen vector.
    /// NB: the tensor owns the allocated memory and as such the tensor is resizable.
    ///
    template <typename tscalar_>
    class tensor_vstorage_t
    {
    public:

        using tscalar = typename std::remove_const<tscalar_>::type;
        using tstorage = tensor_vector_t<tscalar>;
        using treference = tscalar&;
        using tconst_reference = const tscalar&;

        static constexpr bool resizable = true;
        static constexpr bool owns_memory = true;

        tensor_vstorage_t() = default;
        ~tensor_vstorage_t() = default;
        tensor_vstorage_t(tensor_vstorage_t&&) noexcept = default;
        tensor_vstorage_t(const tensor_vstorage_t&) = default;
        tensor_vstorage_t& operator=(tensor_vstorage_t&&) noexcept = default;
        tensor_vstorage_t& operator=(const tensor_vstorage_t&) = default;

        explicit tensor_vstorage_t(const tstorage& data) : m_data(data) {}
        explicit tensor_vstorage_t(const tensor_size_t size) : m_data(size) {}

        template <typename tscalar2>
        explicit tensor_vstorage_t(const tensor_pstorage_t<tscalar2>& other) :
            m_data(map_vector(other.data(), other.size()))
        {
        }

        template <typename tscalar2>
        tensor_vstorage_t& operator=(const tensor_pstorage_t<tscalar2>& other);

        [[nodiscard]] auto size() const { return m_data.size(); }
        void resize(const tensor_size_t size) { m_data.resize(size); }

        auto data() { return m_data.data(); }
        [[nodiscard]] auto data() const { return m_data.data(); }

    private:

        // attributes
        tstorage        m_data;         ///< store tensor as a 1D vector.
    };

    ///
    /// \brief tensor storage using a C-array.
    /// NB: the tensors doesn't own the allocated memory and as such is not resizable.
    ///
    template <typename tscalar_>
    class tensor_pstorage_t
    {
    public:

        using tscalar = typename std::remove_const<tscalar_>::type;
        using tstorage = tscalar_*;
        using treference = typename std::conditional<std::is_const<tscalar_>::value, const tscalar&, tscalar&>::type;
        using tconst_reference = treference;

        static constexpr bool resizable = false;
        static constexpr bool owns_memory = false;

        tensor_pstorage_t() = default;
        ~tensor_pstorage_t() = default;
        tensor_pstorage_t(tensor_pstorage_t&&) noexcept = default;
        tensor_pstorage_t(const tensor_pstorage_t&) = delete;
        tensor_pstorage_t& operator=(tensor_pstorage_t&& other) noexcept
        {
            copy(other);
            return *this;
        }
        tensor_pstorage_t& operator=(const tensor_pstorage_t& other)// NOLINT(cert-oop54-cpp)
        {
            tensor_pstorage_t object = other;
            *this = std::move(object);
            return *this;
        }

        tensor_pstorage_t(const tstorage& data, const tensor_size_t size) :
            m_data(data), m_size(size)
        {
        }

        template <typename tscalar2>
        explicit tensor_pstorage_t(const tensor_vstorage_t<tscalar2>& other) :
            m_data(other.data()), m_size(other.size())
        {
        }

        template <typename tscalar2>
        explicit tensor_pstorage_t(const tensor_pstorage_t<tscalar2>& other) :
            m_data(other.data()), m_size(other.size())
        {
        }

        template <typename tscalar2>
        tensor_pstorage_t& operator=(const tensor_vstorage_t<tscalar2>& other)
        {
            copy(other);
            return *this;
        }

        template <typename tscalar2>
        tensor_pstorage_t& operator=(const tensor_pstorage_t<tscalar2>& other)
        {
            copy(other);
            return *this;
        }

        auto data() { return m_data; }
        [[nodiscard]] auto data() const { return m_data; }
        [[nodiscard]] auto size() const { return m_size; }

    private:

        template <typename tstorage2>
        tensor_pstorage_t& copy(const tstorage2& other)
        {
            assert(size() == other.size());
            map_vector(data(), size()) = map_vector(other.data(), other.size());
            return *this;
        }

        // attributes
        tstorage        m_data{nullptr};///< wrap tensor over a contiguous array.
        tensor_size_t   m_size{0};      ///<
    };

    template <typename tscalar>
    template <typename tscalar2>
    tensor_vstorage_t<tscalar>& tensor_vstorage_t<tscalar>::operator=(const tensor_pstorage_t<tscalar2>& other)
    {
        m_data = map_vector(other.data(), other.size());
        return *this;
    }
}
