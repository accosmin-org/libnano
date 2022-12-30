#pragma once

#include <nano/core/clonable.h>
#include <nano/core/stream.h>

namespace nano
{
    ///
    /// \brief serialize objects registered to factory.
    ///
    template <typename tobject, std::enable_if_t<std::is_base_of_v<clonable_t<tobject>, tobject>, bool> = true>
    std::ostream& write(std::ostream& stream, const std::unique_ptr<tobject>& object)
    {
        if (!::nano::write(stream, object->type_id()) || !::nano::write(stream, *object))
        {
            stream.setstate(std::ios_base::failbit);
        }
        return stream;
    }

    ///
    /// \brief serialize objects registered to factory.
    ///
    template <typename tobject, std::enable_if_t<std::is_base_of_v<clonable_t<tobject>, tobject>, bool> = true>
    std::istream& read(std::istream& stream, std::unique_ptr<tobject>& object)
    {
        string_t type_id;
        if (!::nano::read(stream, type_id))
        {
            stream.setstate(std::ios_base::failbit);
        }

        object = tobject::all().get(type_id);
        if (!object)
        {
            stream.setstate(std::ios_base::failbit);
            return stream;
        }

        return ::nano::read(stream, *object);
    }
} // namespace nano
