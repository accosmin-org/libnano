#pragma once

#include <nano/arch.h>
#include <nano/string.h>

namespace nano
{
///
/// \brief interface for objects that can be identified with an ID,
///     useful within factories to identify specific interface implementations.
///
class NANO_PUBLIC typed_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit typed_t(string_t type_id)
        : m_type_id(std::move(type_id))
    {
    }

    ///
    /// \brief returns the object's type ID.
    ///
    const string_t& type_id() const { return m_type_id; }

protected:
    void rename(string_t type_id) { m_type_id = std::move(type_id); }

private:
    // attributes
    string_t m_type_id; ///< type ID
};
} // namespace nano
