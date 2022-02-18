#pragma once

#include <cassert>
#include <nano/core/logger.h>
#include <nano/core/stream.h>
#include <nano/core/estimator.h>

namespace nano
{
    ///
    /// \brief wraps estimator objects with an associated factory ID
    ///     to support type-safe binary serialization.
    ///
    /// NB: the wrapped object must implement:
    ///     - ::all() to return its associated factory and
    ///     - .clone() to create a copy as stored in its associated factory.
    ///
    template
    <
        typename tobject,
        typename trobject = std::unique_ptr<tobject>,
        std::enable_if_t<std::is_base_of_v<estimator_t, tobject>, bool> = true
    >
    class NANO_PUBLIC identifiable_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        identifiable_t() = default;

        ///
        /// \brief constructor
        ///
        identifiable_t(std::string&& id, trobject&& object) :
            m_id(std::move(id)),
            m_object(std::move(object))
        {
        }

        ///
        /// \brief enable moving
        ///
        identifiable_t(identifiable_t&&) noexcept = default;
        identifiable_t& operator=(identifiable_t&&) noexcept = default;

        ///
        /// \brief enable copying by cloning
        ///
        identifiable_t(const identifiable_t& other) :
            m_id(other.m_id)
        {
            if (other.m_object != nullptr)
            {
                m_object = other.m_object->clone();
            }
        }
        identifiable_t& operator=(const identifiable_t& other)// NOLINT(cert-oop54-cpp)
        {
            auto copy = identifiable_t{other};
            using std::swap;
            swap(m_id, copy.m_id);
            swap(m_object, copy.m_object);
            return *this;
        }

        ///
        /// \brief destructor
        ///
        ~identifiable_t() = default;

        ///
        /// \brief serialize from binary stream.
        ///
        void read(std::istream& stream)
        {
            critical(
                !::nano::read(stream, m_id),
                "identifiable: failed to read from stream!");

            m_object = tobject::all().get(m_id);
            critical(
                m_object == nullptr,
                "identifiable: invalid id <", m_id, "> read from stream!");

            m_object->read(stream);
        }

        ///
        /// \brief serialize to binary stream.
        ///
        void write(std::ostream& stream) const
        {
            critical(
                !::nano::write(stream, m_id),
                "identifiable: failed to write to stream!");

            critical(
                m_object == nullptr,
                "identifiable: cannot serialize unitialized object!");
            m_object->write(stream);
        }

        ///
        /// \brief returns true if the wrapped object is initialized.
        ///
        explicit operator bool() const
        {
            return m_object != nullptr;
        }

        ///
        /// \brief access functions
        ///
        const auto& id() const { return m_id; }
        auto& get() { assert(m_object != nullptr); return *m_object; }
        const auto& get() const { assert(m_object != nullptr); return *m_object; }

    private:

        // attributes
        std::string     m_id;           ///<
        trobject        m_object;       ///<
    };

    template <typename tobject>
    std::istream& read(std::istream& stream, identifiable_t<tobject>& object)
    {
        object.read(stream);
        return stream;
    }

    template <typename tobject>
    std::ostream& write(std::ostream& stream, const identifiable_t<tobject>& object)
    {
        object.write(stream);
        return stream;
    }
}
