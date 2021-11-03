#pragma once

#include <nano/version.h>
#include <nano/core/parameter.h>

namespace nano
{
    ///
    /// \brief interface for serializable object with support for:
    ///     - versioning with automatic checks.
    ///     - integer, scalar or enumeration parameters addressable by name.
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
        /// \brief register new parameters.
        ///
        void register_param(eparam1_t param);
        void register_param(iparam1_t param);
        void register_param(sparam1_t param);

        ///
        /// \brief set parameter values by name.
        ///
        void set(const char* name, int32_t value);
        void set(const char* name, int64_t value);
        void set(const char* name, scalar_t value);

        void set(const string_t& name, int32_t value);
        void set(const string_t& name, int64_t value);
        void set(const string_t& name, scalar_t value);

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        void set(const char* name, tenum value) { find(name).set(value); }

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        void set(const string_t& name, tenum value) { find(name).set(value); }

        ///
        /// \brief retrieve parameter values by name.
        ///
        int64_t ivalue(const char* name) const;
        scalar_t svalue(const char* name) const;

        int64_t ivalue(const string_t& name) const;
        scalar_t svalue(const string_t& name) const;

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        tenum evalue(const char* name) const { return find(name).evalue<tenum>(); }

        template <typename tenum, std::enable_if_t<std::is_enum_v<tenum>, bool> = true>
        tenum evalue(const string_t& name) const { return find(name).evalue<tenum>(); }

        ///
        /// \brief returns all stored parameters.
        ///
        const auto& params() const { return m_parameters; }

        ///
        /// \brief returns the software version.
        ///
        auto major_version() const { return m_major_version; }
        auto minor_version() const { return m_minor_version; }
        auto patch_version() const { return m_patch_version; }

    private:

        parameter_t& find(const char* name);
        parameter_t& find(const string_t& name);

        const parameter_t& find(const char* name) const;
        const parameter_t& find(const string_t& name) const;

        // attributes
        int32_t         m_major_version{::nano::major_version}; ///<
        int32_t         m_minor_version{::nano::minor_version}; ///<
        int32_t         m_patch_version{::nano::patch_version}; ///<
        parameters_t    m_parameters;                           ///<
    };

    NANO_PUBLIC std::istream& read(std::istream& stream, serializable_t&);
    NANO_PUBLIC std::ostream& write(std::ostream& stream, const serializable_t&);
}
