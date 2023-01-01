#pragma once

#include <algorithm>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

namespace nano
{
    template <typename tscalar,
              std::enable_if_t<std::is_standard_layout_v<tscalar> && std::is_trivial_v<tscalar>, bool> = true>
    std::ostream& write(std::ostream& stream, tscalar scalar)
    {
        return stream.write(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<const char*>(&scalar), static_cast<std::streamsize>(sizeof(tscalar)));
    }

    template <typename tscalar, typename tcount,
              std::enable_if_t<std::is_standard_layout_v<tscalar> && std::is_trivial_v<tscalar>, bool> = true>
    std::ostream& write(std::ostream& stream, const tscalar* data, const tcount count)
    {
        return stream.write(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(sizeof(tscalar)) * static_cast<std::streamsize>(count));
    }

    template <typename twscalar, typename tscalar, typename tcount,
              std::enable_if_t<std::is_standard_layout_v<tscalar> && std::is_trivial_v<tscalar>, bool> = true>
    std::ostream& write_cast(std::ostream& stream, const tscalar* data, const tcount count)
    {
        for (tcount i = 0; i < count; ++i)
        {
            write(stream, static_cast<twscalar>(data[i]));
        }
        return stream;
    }

    inline std::ostream& write(std::ostream& stream, const std::string& string)
    {
        write(stream, static_cast<uint32_t>(string.size()));
        return write(stream, string.data(), string.size());
    }

    ///
    /// \brief serialize objects with a write method to binary stream.
    ///
    template <typename tobject,
              std::enable_if_t<std::is_member_function_pointer_v<decltype(&tobject::write)>, bool> = true>
    std::ostream& write(std::ostream& stream, const tobject& object)
    {
        return object.write(stream);
    }

    ///
    /// \brief serialize factory objects to binary stream.
    ///
    template <typename tobject>
    std::ostream& write(std::ostream& stream, const std::unique_ptr<tobject>& object)
    {
        if (!::nano::write(stream, object->type_id()) || !::nano::write(stream, *object))
        {
            stream.setstate(std::ios_base::failbit);
        }
        return stream;
    }

    ///
    /// \brief serialize a vector of values to binary stream.
    ///
    template <typename tvalue>
    std::ostream& write(std::ostream& stream, const std::vector<tvalue>& values)
    {
        if (!write(stream, static_cast<uint64_t>(values.size())))
        {
            return stream;
        }

        [[maybe_unused]] const auto ret = std::any_of(values.begin(), values.end(),
                                                      [&](const auto& value)
                                                      {
                                                          return !write(stream, value); // LCOV_EXCL_LINE
                                                      });
        return stream;
    }

    template <typename tscalar,
              std::enable_if_t<std::is_standard_layout_v<tscalar> && std::is_trivial_v<tscalar>, bool> = true>
    std::istream& read(std::istream& stream, tscalar& scalar)
    {
        return stream.read(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<char*>(&scalar), static_cast<std::streamsize>(sizeof(tscalar)));
    }

    template <typename tscalar, typename tcount,
              std::enable_if_t<std::is_standard_layout_v<tscalar> && std::is_trivial_v<tscalar>, bool> = true>
    std::istream& read(std::istream& stream, tscalar* data, tcount count)
    {
        return stream.read(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            reinterpret_cast<char*>(data),
            static_cast<std::streamsize>(sizeof(tscalar)) * static_cast<std::streamsize>(count));
    }

    template <typename trscalar, typename tscalar,
              std::enable_if_t<std::is_standard_layout_v<tscalar> && std::is_trivial_v<tscalar>, bool> = true>
    std::istream& read_cast(std::istream& stream, tscalar& scalar)
    {
        trscalar value{};
        read(stream, value);
        scalar = static_cast<tscalar>(value);
        return stream;
    }

    template <typename trscalar, typename tscalar, typename tcount,
              std::enable_if_t<std::is_standard_layout_v<tscalar> && std::is_trivial_v<tscalar>, bool> = true>
    std::istream& read_cast(std::istream& stream, tscalar* data, const tcount count)
    {
        for (tcount i = 0; i < count; ++i)
        {
            read_cast<trscalar>(stream, data[i]);
        }
        return stream;
    }

    inline std::istream& read(std::istream& stream, std::string& string)
    {
        uint32_t size = 0;
        if (!read(stream, size))
        {
            return stream;
        }

        string.resize(size);
        for (char& c : string)
        {
            read(stream, c);
        }
        return stream;
    }

    ///
    /// \brief serialize objects with a read method from binary stream.
    ///
    template <typename tobject,
              std::enable_if_t<std::is_member_function_pointer_v<decltype(&tobject::read)>, bool> = true>
    std::istream& read(std::istream& stream, tobject& object)
    {
        return object.read(stream);
    }

    ///
    /// \brief serialize factory objects from binary stream.
    ///
    template <typename tobject>
    std::istream& read(std::istream& stream, std::unique_ptr<tobject>& object)
    {
        std::string type_id;
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

    ///
    /// \brief serialize a vector of values from binary stream.
    ///
    template <typename tvalue>
    std::istream& read(std::istream& stream, std::vector<tvalue>& values)
    {
        uint64_t size = 0;
        if (!read(stream, size))
        {
            return stream;
        }

        values.resize(size);
        for (auto& value : values)
        {
            if (!read(stream, value))
            {
                return stream; // LCOV_EXCL_LINE
            }
        }
        return stream;
    }
} // namespace nano
