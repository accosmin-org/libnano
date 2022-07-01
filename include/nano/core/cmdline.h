#pragma once

#include <iostream>
#include <nano/arch.h>
#include <nano/core/strutil.h>
#include <unordered_map>

namespace nano
{
    ///
    /// \brief command line processing of the form:
    ///     --option [value]
    ///     -o [value]
    ///     --additional-option [value]
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
        /// \brief command line option.
        ///
        struct option_t
        {
            option_t();
            option_t(string_t short_name, string_t name, string_t description, string_t default_value);

            string_t describe() const;

            // attributes
            string_t m_short_name;    ///<
            string_t m_name;          ///<
            string_t m_description;   ///<
            string_t m_default_value; ///<
        };

        using options_t = std::vector<option_t>;

        ///
        /// \brief result of parsing the command line.
        ///
        struct result_t
        {
            using storage_t = std::unordered_map<string_t, string_t>;

            bool     has(const string_t& option_name) const;
            string_t get(const string_t& option_name) const;

            template <typename tvalue>
            tvalue get(const string_t& option_name) const
            {
                return nano::from_string<tvalue>(get(option_name));
            }

            // attributes
            storage_t m_ovalues; ///< values for known short or long options
            storage_t m_xvalues; ///< values for additional (extra) options
        };

        ///
        /// \brief constructor
        ///
        explicit cmdline_t(string_t title);

        ///
        /// \brief add new option by name and short name (without dash).
        ///
        void add(string_t short_name, string_t name, string_t description);

        ///
        /// \brief add new option with default value by name and short name (without dash).
        ///
        template <typename tvalue>
        void add(string_t short_name, string_t name, string_t description, tvalue default_value)
        {
            add(std::move(short_name), std::move(name), std::move(description), scat(default_value));
        }

        ///
        /// \brief process the command line arguments.
        ///
        result_t process(int argc, const char* argv[]) const;

        ///
        /// \brief process the command line arguments.
        ///
        result_t process(const string_t& config) const;

        ///
        /// \brief process the command line arguments from configuration file.
        ///
        result_t process_config_file(const string_t& path) const;

        ///
        /// \brief print help menu.
        ///
        void usage(std::ostream& = std::cout, size_t indent = 2) const;

    private:
        void add(string_t short_name, string_t name, string_t description, string_t default_value);

        // attributes
        string_t  m_title;   ///<
        options_t m_options; ///<
    };
} // namespace nano
