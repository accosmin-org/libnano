#pragma once

#include <nano/core/parameter.h>
#include <nano/version.h>

namespace nano
{
///
/// \brief interface for configurable objects with support for:
///     - versioning with automatic checks.
///     - constrained parameters addressable by name.
///     - serialization to and from binary streams.
///
class NANO_PUBLIC configurable_t
{
public:
    ///
    /// \brief default constructor
    ///
    configurable_t() = default;

    ///
    /// \brief enable copying
    ///
    configurable_t(const configurable_t&)            = default;
    configurable_t& operator=(const configurable_t&) = default;

    ///
    /// \brief enable moving
    ///
    configurable_t(configurable_t&&) noexcept            = default;
    configurable_t& operator=(configurable_t&&) noexcept = default;

    ///
    /// \brief default destructor
    ///
    virtual ~configurable_t() = default;

    ///
    /// \brief serialize from the given binary stream.
    ///
    /// NB: any error is considered critical and expected to result in an exception.
    ///
    virtual std::istream& read(std::istream&);

    ///
    /// \brief serialize to the given binary stream.
    ///
    /// NB: any error is considered critical and expected to result in an exception.
    ///
    virtual std::ostream& write(std::ostream&) const;

    ///
    /// \brief register a new parameter if possible, otherwise throw an exception.
    ///
    void register_parameter(parameter_t);

    ///
    /// \brief return the parameter with the given name if any, otherwise return nullptr.
    ///
    parameter_t* parameter_if(const char* name);
    parameter_t* parameter_if(const string_t& name);

    const parameter_t* parameter_if(const char* name) const;
    const parameter_t* parameter_if(const string_t& name) const;

    ///
    /// \brief return the parameter with the given name if any, otherwise throw an exception.
    ///
    parameter_t& parameter(const char* name);
    parameter_t& parameter(const string_t& name);

    const parameter_t& parameter(const char* name) const;
    const parameter_t& parameter(const string_t& name) const;

    ///
    /// \brief returns all stored parameters.
    ///
    const auto& parameters() const { return m_parameters; }

    ///
    /// \brief returns the software's major version.
    ///
    auto major_version() const { return m_major_version; }

    ///
    /// \brief returns the software's minor version.
    ///
    auto minor_version() const { return m_minor_version; }

    ///
    /// \brief returns the software's patch version.
    ///
    auto patch_version() const { return m_patch_version; }

private:
    // attributes
    int32_t      m_major_version{::nano::major_version}; ///<
    int32_t      m_minor_version{::nano::minor_version}; ///<
    int32_t      m_patch_version{::nano::patch_version}; ///<
    parameters_t m_parameters;                           ///<
};
} // namespace nano
