#include <cassert>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <nano/cmdline.h>

using namespace nano;

struct option_t
{
    explicit option_t(string_t short_name = string_t(),
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

bool operator==(const option_t& option, const string_t& name_or_short_name)
{
    return  option.m_short_name == name_or_short_name ||
            option.m_name == name_or_short_name;
}

using options_t = std::vector<option_t>;

struct cmdline_t::impl_t
{
    explicit impl_t(string_t title) :
        m_title(std::move(title))
    {
    }

    auto find(const string_t& name_or_short_name)
    {
        return std::find(m_options.begin(), m_options.end(), name_or_short_name);
    }

    auto store(const string_t& name_or_short_name, const string_t& value = string_t())
    {
        auto it = find(name_or_short_name);
        if (it == m_options.end())
        {
            log_critical("cmdline: unrecognized option [" + name_or_short_name + "]");
        }
        else
        {
            it->m_given = true;
            it->m_value = value;
        }
    }

    void log_critical(const string_t& message) const
    {
        std::cout << message << std::endl << std::endl;
        usage();
    }

    [[noreturn]] void usage() const
    {
        std::cout << m_title << std::endl;

        size_t max_option_size = 0;
        for (const auto& option : m_options)
        {
            max_option_size = std::max(max_option_size, option.concatenate().size());
        }

        max_option_size += 4;
        for (const auto& option : m_options)
        {
            std::cout << "  " << nano::align(option.concatenate(), max_option_size)
                  << option.m_description << std::endl;
        }

        std::cout << std::endl;
        exit(EXIT_FAILURE);
    }

    string_t        m_title;
    options_t       m_options;
};

cmdline_t::cmdline_t(const string_t& title) :
    m_impl(new impl_t(title))
{
    add("h", "help", "usage");
}

cmdline_t::~cmdline_t() = default;

void cmdline_t::add(const string_t& short_name, const string_t& name, const string_t& description) const
{
    const string_t default_value;
    add(short_name, name, description, default_value);
}

void cmdline_t::add(
    const string_t& short_name, const string_t& name, const string_t& description,
    const string_t& default_value) const
{
    if (name.empty() ||
        nano::starts_with(name, "-") ||
        nano::starts_with(name, "--"))
    {
        log_critical("cmdline: invalid option name [" + name + "]");
    }

    if (!short_name.empty() &&
        (short_name.size() != 1 || short_name[0] == '-'))
    {
        log_critical("cmdline: invalid short option name [" + short_name + "]");
    }

    if (m_impl->find(name) != m_impl->m_options.end())
    {
        log_critical("cmdline: duplicated option [" + name + "]");
    }

    if (!short_name.empty() &&
        m_impl->find(short_name) != m_impl->m_options.end())
    {
        log_critical("cmdline: duplicated option [" + short_name + "]");
    }

    m_impl->m_options.emplace_back(short_name, name, description, default_value);
}

void cmdline_t::process(const int argc, const char* argv[]) const
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
                log_critical(strcat("cmdline: invalid option name [", name, "/", token, "]"));
            }

            m_impl->store(name);
            current_name_or_short_name = name;
        }
        else if (nano::starts_with(token, "-"))
        {
            const string_t short_name = token.substr(1);

            if (short_name.size() != 1)
            {
                log_critical(strcat("cmdline: invalid short option name [", short_name, "/", token, "]"));
            }

            m_impl->store(short_name);
            current_name_or_short_name = short_name;
        }
        else
        {
            const string_t& value = token;

            if (current_name_or_short_name.empty())
            {
                log_critical(strcat("cmdline: missing option before value [", value, "]"));
            }

            m_impl->store(current_name_or_short_name, value);
            current_name_or_short_name.clear();
        }
    }

    if (has("help"))
    {
        usage();
    }
}

void cmdline_t::process(const string_t& config) const
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

void cmdline_t::process_config_file(const string_t& path) const
{
    std::ifstream in(path.c_str());
    const string_t config((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    process(config);
}

bool cmdline_t::has(const string_t& name_or_short_name) const
{
    const auto it = m_impl->find(name_or_short_name);
    if (it == m_impl->m_options.end())
    {
        log_critical("cmdline: unrecognized option [" + name_or_short_name + "]");
    }
    return it->m_given;
}

string_t cmdline_t::get(const string_t& name_or_short_name) const
{
    const auto it = m_impl->find(name_or_short_name);
    if (it == m_impl->m_options.end())
    {
        log_critical("cmdline: unrecognized option [" + name_or_short_name + "]");
    }
    else if (!it->m_given && it->m_default_value.empty())
    {
        log_critical("cmdline: no value provided for option [" + name_or_short_name + "]");
    }
    return it->get();
}

void cmdline_t::log_critical(const string_t& message) const
{
    m_impl->log_critical(message);
}

void cmdline_t::usage() const
{
    m_impl->usage();
}
