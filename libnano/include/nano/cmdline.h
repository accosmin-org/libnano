#pragma once

#include <memory>
#include <nano/arch.h>
#include <nano/string_utils.h>

namespace nano
{
    ///
    /// \brief command line processing of the form:
    ///     --option [value]
    ///     -o [value]s
    ///
    /// other properties:
    ///     - -h,--help is built-in
    ///     - any error is considered critical and reported as an exception
    ///             (e.g. duplicated option names, missing option value, invalid option name)
    ///     - each option must have a long name, while the short name (single character) is optional
    ///     - options need not have an associated value (they can be interpreted as boolean flags)
    ///
    class NANO_PUBLIC cmdline_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit cmdline_t(const string_t& title);

        ///
        /// \brief disable copying
        ///
        cmdline_t(const cmdline_t&) = delete;
        cmdline_t& operator=(const cmdline_t&) = delete;

        ///
        /// \brief destructor
        ///
        ~cmdline_t();

        ///
        /// \brief add new option by name and short name (without dash)
        ///
        void add(const string_t& short_name, const string_t& name, const string_t& description) const;

        ///
        /// \brief add new option with default value by name and short name (without dash)
        ///
        template <typename tvalue>
        void add(const string_t& short_name, const string_t& name, const string_t& description,
            const tvalue default_value) const
        {
            add(short_name, name, description, to_string(default_value));
        }

        ///
        /// \brief process the command line arguments
        ///
        void process(const int argc, const char* argv[]) const;

        ///
        /// \brief process the command line arguments
        ///
        void process(const string_t& config) const;

        ///
        /// \brief process the command line arguments from configuration file
        ///
        void process_config_file(const string_t& path) const;

        ///
        /// \brief check if an option was set
        ///
        bool has(const string_t& name_or_short_name) const;

        ///
        /// \brief get the value of an option
        ///
        string_t get(const string_t& name_or_short_name) const;

        ///
        /// \brief get the value of an option as a given type
        ///
        template <typename tvalue>
        tvalue get(const string_t& name_or_short_name) const
        {
            return nano::from_string<tvalue>(get(name_or_short_name));
        }

        ///
        /// \brief print help menu
        ///
        void usage() const;

    private:

        void log_critical(const string_t& message) const;

        ///
        /// \brief add a new option
        ///
        void add(const string_t& short_name, const string_t& name, const string_t& description,
                 const string_t& default_value) const;

        // attributes
        struct impl_t;
        std::unique_ptr<impl_t> m_impl;
    };
}

