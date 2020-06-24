#pragma once

#include <string>
#include <istream>
#include <ostream>
#include <type_traits>
#include <nano/arch.h>
#include <nano/version.h>

namespace nano
{
    namespace detail
    {
        template
        <
            typename tscalar,
            typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
        >
        // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
        std::ostream& write(std::ostream& stream, const tscalar scalar)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            return stream.write(reinterpret_cast<const char*>(&scalar), sizeof(tscalar));
        }

        template
        <
            typename tscalar, typename tcount,
            typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
        >
        std::ostream& write(std::ostream& stream, const tscalar* data, const tcount count)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            return stream.write(reinterpret_cast<const char*>(data), sizeof(tscalar) * count);
        }

        template
        <
            typename twscalar, typename tscalar, typename tcount,
            typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
        >
        std::ostream& write_cast(std::ostream& stream, const tscalar* data, const tcount count)
        {
            for (tcount i = 0; i < count; ++ i)
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

        template
        <
            typename tscalar,
            typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
        >
        std::istream& read(std::istream& stream, tscalar& scalar)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            return stream.read(reinterpret_cast<char*>(&scalar), sizeof(tscalar));
        }

        template
        <
            typename tscalar, typename tcount,
            typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
        >
        std::istream& read(std::istream& stream, tscalar* data, const tcount count)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            return stream.read(reinterpret_cast<char*>(data), sizeof(tscalar) * count);
        }

        template
        <
            typename trscalar, typename tscalar,
            typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
        >
        std::istream& read_cast(std::istream& stream, tscalar& scalar)
        {
            trscalar value{};
            read(stream, value);
            scalar = static_cast<tscalar>(value);
            return stream;
        }

        template
        <
            typename trscalar, typename tscalar, typename tcount,
            typename = typename std::enable_if<std::is_pod<tscalar>::value>::type
        >
        std::istream& read_cast(std::istream& stream, tscalar* data, const tcount count)
        {
            for (tcount i = 0; i < count; ++ i)
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
    }

    ///
    /// \brief interface for serializable object with support for:
    ///     - versioning.
    ///
    class NANO_PUBLIC serializable_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        serializable_t() = default;

        ///
        /// \brief enable copying
        ///
        serializable_t(const serializable_t&) = default;
        serializable_t& operator=(const serializable_t&) = default;

        ///
        /// \brief enable moving
        ///
        serializable_t(serializable_t&&) noexcept = default;
        serializable_t& operator=(serializable_t&&) noexcept = default;

        ///
        /// \brief default destructor
        ///
        virtual ~serializable_t() = default;

        ///
        /// \brief serialize from the given binary stream.
        ///
        /// NB: any error is considered critical and expected to result in an exception.
        ///
        virtual void read(std::istream&);

        ///
        /// \brief serialize to the given binary stream.
        ///
        /// NB: any error is considered critical and expected to result in an exception.
        ///
        virtual void write(std::ostream&) const;

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto major_version() const { return m_major_version; }
        [[nodiscard]] auto minor_version() const { return m_minor_version; }
        [[nodiscard]] auto patch_version() const { return m_patch_version; }

    private:

        // attributes
        int32_t     m_major_version{::nano::major_version}; ///<
        int32_t     m_minor_version{::nano::minor_version}; ///<
        int32_t     m_patch_version{::nano::patch_version}; ///<
    };
}
