#pragma once

#include <cassert>
#include <filesystem>
#include <iostream>
#include <limits>
#include <nano/configurable.h>
#include <nano/logger.h>
#include <optional>
#include <unordered_map>

namespace nano
{
using ostring_t = std::optional<string_t>;

///
/// \brief command line option consisting of:
///     - comma-separated keywords starting with single or double dashes (e.g. -h,--help),
///     - description and
///     - an optional default value.
///
struct NANO_PUBLIC cmdoption_t
{
    // attributes
    string_t  m_keywords;      ///<
    string_t  m_description;   ///<
    ostring_t m_default_value; ///<
};

using cmdoptions_t = std::vector<cmdoption_t>;

///
/// \brief command line option value consisting of:
///     - optional value and
///     - index in the list of options.
///
struct NANO_PUBLIC cmdvalue_t
{
    static constexpr auto no_index = std::numeric_limits<size_t>::max();

    bool is_extra() const { return m_index == no_index; }

    bool has_value() const { return m_value.has_value(); }

    string_t value() const
    {
        assert(has_value());
        return m_value.value(); // NOLINT(bugprone-unchecked-optional-access)
    }

    // attributes
    ostring_t m_value;           ///<
    size_t    m_index{no_index}; ///<
};

using cmdvalues_t = std::unordered_map<string_t, cmdvalue_t>;

///
/// \brief the result of parsing command line arguments consisting of:
///     - a set that maps option names to optional values.
///
/// NB: the unknown option names are stored as they can be useful to setup configurable objects at runtime.
///
struct NANO_PUBLIC cmdresult_t
{
    cmdresult_t();
    explicit cmdresult_t(const cmdvalues_t&);

    bool     has(std::string_view option_name) const;
    string_t get(std::string_view option_name) const;
    bool     has_value(std::string_view option_name) const;

    template <class tvalue>
    tvalue get(const std::string_view option_name) const
    {
        return ::nano::from_string<tvalue>(get(option_name));
    }

    // attributes
    cmdvalues_t m_values; ///<
};

///
/// \brief RAII utility to keep track of the used parameters and
///     log all unused parameters at the end (e.g. typos, not matching to any solver).
///
class NANO_PUBLIC cmdconfig_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit cmdconfig_t(const cmdresult_t&, logger_t logger = make_stdout_logger());

    ///
    /// \brief disable copying and moving.
    ///
    cmdconfig_t(cmdconfig_t&&)                 = delete;
    cmdconfig_t(const cmdconfig_t&)            = delete;
    cmdconfig_t& operator=(cmdconfig_t&&)      = delete;
    cmdconfig_t& operator=(const cmdconfig_t&) = delete;

    ///
    /// \brief configurable the given object and update the list of used parameters.
    ///
    void setup(configurable_t&);

    ///
    /// \brief destructor (log the unused parameters).
    ///
    ~cmdconfig_t();

private:
    using storage_t = std::unordered_map<string_t, int>;

    // attributes
    const cmdresult_t& m_options;      ///<
    logger_t           m_logger;       ///<
    storage_t          m_params_usage; ///<
};

///
/// \brief command line processing of the form:
///     --option [value]
///     -o [value]
///     --additional-option [value]
///
/// other properties:
///     - the option `-h,--help` is built-in: prints the detailed usage
///     - the option `-v,--version` is built-in: prints the library version
///     - the option `-g,--git-hash` is built-in: prints the library's git commit hash
///     - any error is considered critical and reported as an exception
///          (e.g. duplicated option names, missing option value, invalid option name)
///     - options need not have an associated value (they can be interpreted as boolean flags)
///     - additional unregistered options are supported (e.g. to set parameters of configurable objects)
///
class NANO_PUBLIC cmdline_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit cmdline_t(string_t title);

    ///
    /// \brief register a new option, optionally with a default value.
    ///
    void add(cmdoption_t);
    void add(string_t keywords, string_t description);
    void add(string_t keywords, string_t description, string_t default_value);

    template <class tvalue>
    void add(string_t keywords, string_t description, tvalue default_value)
    {
        add(std::move(keywords), std::move(description), scat(default_value));
    }

    ///
    /// \brief process the command line arguments.
    ///
    cmdresult_t process(int argc, const char* argv[]) const;

    ///
    /// \brief process the command line arguments.
    ///
    cmdresult_t process(const string_t& config) const;

    ///
    /// \brief process the command line arguments from configuration file.
    ///
    cmdresult_t process_config_file(const std::filesystem::path& path) const;

    ///
    /// \brief handle the builtin arguments of the given processed command line arguments
    ///     (e.g. usage with -h,--help, library version with -v,--version).
    ///
    bool handle(const cmdresult_t&, std::ostream& = std::cout, size_t indent = 2) const;

private:
    void usage(std::ostream&, size_t indent) const;

    // attributes
    string_t     m_title;   ///<
    cmdoptions_t m_options; ///<
    cmdvalues_t  m_values;  ///<
};
} // namespace nano
