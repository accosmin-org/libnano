#pragma once

#include <nano/parameter.h>
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
    configurable_t& operator=(const configurable_t&) = default; // LCOV_EXCL_LINE

    ///
    /// \brief enable moving
    ///
    configurable_t(configurable_t&&) noexcept            = default;
    configurable_t& operator=(configurable_t&&) noexcept = default; // LCOV_EXCL_LINE

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
    parameter_t*       parameter_if(std::string_view name);
    const parameter_t* parameter_if(std::string_view name) const;

    ///
    /// \brief return the parameter with the given name if any, otherwise throw an exception.
    ///
    parameter_t&       parameter(std::string_view name);
    const parameter_t& parameter(std::string_view name) const;

    ///
    /// \brief returns all stored parameters.
    ///
    const parameters_t& parameters() const { return m_parameters; }

    ///
    /// \brief returns the software's major version.
    ///
    int32_t major_version() const { return m_major_version; }

    ///
    /// \brief returns the software's minor version.
    ///
    int32_t minor_version() const { return m_minor_version; }

    ///
    /// \brief returns the software's patch version.
    ///
    int32_t patch_version() const { return m_patch_version; }

    ///
    /// \brief configure the object with the given pairs of parameter names and values.
    ///
    configurable_t& config() { return *this; }

    template <class targ, class... targs>
    configurable_t& config(const char* const param_name, const targ value, const targs... args)
    {
        parameter(param_name) = value;
        return config(args...);
    }

private:
    // attributes
    int32_t      m_major_version{::nano::major_version}; ///<
    int32_t      m_minor_version{::nano::minor_version}; ///<
    int32_t      m_patch_version{::nano::patch_version}; ///<
    parameters_t m_parameters;                           ///<
};
} // namespace nano
