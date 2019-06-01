#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <nano/string_utils.h>

namespace nano
{
    ///
    /// \brief command line option.
    ///
    struct option_t
    {
        explicit option_t(
            string_t short_name = string_t(),
            string_t name = string_t(),
            string_t description = string_t(),
            string_t default_value = string_t()) :
            m_short_name(std::move(short_name)),
            m_name(std::move(name)),
            m_description(std::move(description)),
            m_default_value(std::move(default_value))
        {
        }

        string_t concatenate() const
        {
            return  (m_short_name.empty() ? "" : ("-" + m_short_name) + ",") +
                    "--" + m_name +
                    (m_default_value.empty() ? "" : ("(" + m_default_value + ")"));
        }

        bool has() const
        {
            return m_given;
        }

        string_t get() const
        {
            return m_value.empty() ? m_default_value : m_value;
        }

        // attributes
        string_t    m_short_name;
        string_t    m_name;
        string_t    m_description;
        string_t    m_default_value;
        string_t    m_value;
        bool        m_given{false};
    };

    inline bool operator==(const option_t& option, const string_t& name_or_short_name)
    {
        return  option.m_short_name == name_or_short_name ||
                option.m_name == name_or_short_name;
    }

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
    class cmdline_t
    {
    public:

        ///
        /// \brief constructor
        ///
        explicit cmdline_t(const string_t& title);

        ///
        /// \brief add new option by name and short name (without dash)
        ///
        void add(const string_t& short_name, const string_t& name, const string_t& description)
        {
            add(short_name, name, description, string_t());
        }

        ///
        /// \brief add new option with default value by name and short name (without dash)
        ///
        template <typename tvalue>
        void add(const string_t& short_name, const string_t& name, const string_t& description,
            const tvalue default_value)
        {
            add(short_name, name, description, to_string(default_value));
        }

        ///
        /// \brief process the command line arguments
        ///
        void process(const int argc, const char* argv[]);

        ///
        /// \brief process the command line arguments
        ///
        void process(const string_t& config);

        ///
        /// \brief process the command line arguments from configuration file
        ///
        void process_config_file(const string_t& path);

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
        void usage(std::ostream& = std::cout) const;

    private:

        auto find(const string_t& name_or_short_name)
        {
            return std::find(m_options.begin(), m_options.end(), name_or_short_name);
        }

        auto find(const string_t& name_or_short_name) const
        {
            return std::find(m_options.begin(), m_options.end(), name_or_short_name);
        }

        void add(const string_t& short_name, const string_t& name, const string_t& description,
            const string_t& default_value)
        {
            if (name.empty() ||
                nano::starts_with(name, "-") ||
                nano::starts_with(name, "--"))
            {
                throw std::runtime_error("cmdline: invalid option name [" + name + "]");
            }

            if (!short_name.empty() &&
                (short_name.size() != 1 || short_name[0] == '-'))
            {
                throw std::runtime_error("cmdline: invalid short option name [" + short_name + "]");
            }

            if (find(name) != m_options.end())
            {
                throw std::runtime_error("cmdline: duplicated option [" + name + "]");
            }

            if (!short_name.empty() &&
                find(short_name) != m_options.end())
            {
                throw std::runtime_error("cmdline: duplicated option [" + short_name + "]");
            }

            m_options.emplace_back(short_name, name, description, default_value);
        }

        void store(const string_t& name_or_short_name, const string_t& value = string_t())
        {
            auto it = find(name_or_short_name);
            if (it == m_options.end())
            {
                throw std::runtime_error("cmdline: unrecognized option [" + name_or_short_name + "]");
            }
            else
            {
                it->m_given = true;
                it->m_value = value;
            }
        }

        // attributes
        using options_t = std::vector<option_t>;
        string_t        m_title;        ///<
        options_t       m_options;      ///<
    };

    inline cmdline_t::cmdline_t(const string_t& title) :
        m_title(std::move(title))
    {
        add("h", "help", "usage");
    }

    inline void cmdline_t::process(const int argc, const char* argv[])
    {
        string_t current_name_or_short_name;

        for (int i = 1; i < argc; ++ i)
        {
            const string_t token = argv[i];
            assert(!token.empty());

            if (nano::starts_with(token, "--"))
            {
                const string_t name = token.substr(2);
                if (name.empty())
                {
                    throw std::runtime_error(strcat("cmdline: invalid option name [", name, "/", token, "]"));
                }

                store(name);
                current_name_or_short_name = name;
            }
            else if (nano::starts_with(token, "-"))
            {
                const string_t short_name = token.substr(1);
                if (short_name.size() != 1)
                {
                    throw std::runtime_error(strcat("cmdline: invalid short option name [", short_name, "/", token, "]"));
                }

                store(short_name);
                current_name_or_short_name = short_name;
            }
            else
            {
                const string_t& value = token;
                if (current_name_or_short_name.empty())
                {
                    throw std::runtime_error(strcat("cmdline: missing option before value [", value, "]"));
                }

                store(current_name_or_short_name, value);
                current_name_or_short_name.clear();
            }
        }

        if (has("help"))
        {
            usage();
        }
    }

    inline void cmdline_t::process(const string_t& config)
    {
        const auto tokens = nano::split(config, " \t\n\r");

        std::vector<const char*> ptokens(1 + tokens.size());
        ptokens[0] = nullptr;
        for (size_t i = 0; i < tokens.size(); ++ i)
        {
            ptokens[i + 1] = tokens[i].data();
        }

        process(static_cast<int>(tokens.size() + 1), ptokens.data());
    }

    inline void cmdline_t::process_config_file(const string_t& path)
    {
        std::ifstream in(path.c_str());
        const string_t config((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

        process(config);
    }

    inline bool cmdline_t::has(const string_t& name_or_short_name) const
    {
        const auto it = find(name_or_short_name);
        if (it == m_options.end())
        {
            throw std::runtime_error("cmdline: unrecognized option [" + name_or_short_name + "]");
        }
        return it->m_given;
    }

    inline string_t cmdline_t::get(const string_t& name_or_short_name) const
    {
        const auto it = find(name_or_short_name);
        if (it == m_options.end())
        {
            throw std::runtime_error("cmdline: unrecognized option [" + name_or_short_name + "]");
        }
        else if (!it->m_given && it->m_default_value.empty())
        {
            throw std::runtime_error("cmdline: no value provided for option [" + name_or_short_name + "]");
        }
        return it->get();
    }

    inline void cmdline_t::usage(std::ostream& os) const
    {
        os << m_title << std::endl;

        size_t max_option_size = 0;
        for (const auto& option : m_options)
        {
            max_option_size = std::max(max_option_size, option.concatenate().size());
        }

        max_option_size += 4;
        for (const auto& option : m_options)
        {
            os << "  " << nano::align(option.concatenate(), max_option_size)
                  << option.m_description << std::endl;
        }

        os << std::endl;
    }
}
