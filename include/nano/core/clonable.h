#pragma once

#include <memory>
#include <nano/string.h>

namespace nano
{
    ///
    /// \brief interface for objects that can be:
    ///     - copied exactly and
    ///     - identified with a type ID (useful within factories) to identify the specific implementation.
    ///
    template <typename tobject>
    class clonable_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit clonable_t(string_t type_id)
            : m_type_id(std::move(type_id))
        {
        }

        ///
        /// \brief enable copying
        ///
        clonable_t(const clonable_t&)            = default;
        clonable_t& operator=(const clonable_t&) = default;

        ///
        /// \brief enable moving
        ///
        clonable_t(clonable_t&&) noexcept            = default;
        clonable_t& operator=(clonable_t&&) noexcept = default;

        ///
        /// \brief destructor
        ///
        virtual ~clonable_t() = default;

        ///
        /// \brief returns a copy of the current object.
        ///
        virtual std::unique_ptr<tobject> clone() const = 0;

        ///
        /// \brief returns the object's type ID.
        ///
        const auto& type_id() const { return m_type_id; }

    private:
        // attributes
        string_t m_type_id; ///< type ID
    };
} // namespace nano
