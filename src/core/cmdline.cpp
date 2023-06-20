#include <cassert>
#include <fstream>
#include <nano/core/cmdline.h>
#include <nano/core/tokenizer.h>

using namespace nano;

namespace
{
auto find(const cmdline_t::options_t& options, const string_t& name_or_short_name)
{
    return std::find_if(options.begin(), options.end(),
                        [&](const cmdline_t::option_t& option)
                        { return option.m_short_name == name_or_short_name || option.m_name == name_or_short_name; });
}
} // namespace

static const auto str_dash      = string_t{"-"};  // NOLINT(cert-err58-cpp)
static const auto str_dash_dash = string_t{"--"}; // NOLINT(cert-err58-cpp)

cmdline_t::option_t::option_t() = default; // LCOV_EXCL_LINE

cmdline_t::option_t::option_t(string_t short_name, string_t name, string_t description, string_t default_value)
    : m_short_name(std::move(short_name))
    , m_name(std::move(name))
    , m_description(std::move(description))
    , m_default_value(std::move(default_value))
{
}

string_t cmdline_t::option_t::describe() const
{
    string_t str;
    if (!m_short_name.empty())
    {
        str += "-";
        str += m_short_name;
        str += ",";
    }
    str += "--";
    str += m_name;
    if (!m_default_value.empty())
    {
        str += "(";
        str += m_default_value;
        str += ")";
    }

    return str;
} // LCOV_EXCL_LINE

bool cmdline_t::result_t::has(const string_t& name) const
{
    return m_ovalues.find(name) != m_ovalues.end();
}

string_t cmdline_t::result_t::get(const string_t& name) const
{
    const auto it = m_ovalues.find(name);
    if (it == m_ovalues.end())
    {
        throw std::runtime_error(scat("cmdline: unrecognized option [", name, "]"));
    }
    else if (it->second.empty())
    {
        throw std::runtime_error(scat("cmdline: no value provided for option [", name, "]"));
    }
    return it->second;
}

cmdline_t::cmdline_t(string_t title)
    : m_title(std::move(title))
{
    add("h", "help", "usage");
    add("v", "version", "library version");
    add("g", "git-hash", "git commit hash");
}

void cmdline_t::add(string_t short_name, string_t name, string_t description)
{
    add(std::move(short_name), std::move(name), std::move(description), string_t());
}

void cmdline_t::add(string_t short_name, string_t name, string_t description, string_t default_value)
{
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    if (name.empty() || nano::starts_with(name, str_dash_dash) || nano::starts_with(name, str_dash))
    {
        throw std::runtime_error(scat("cmdline: invalid option name [", name, "]"));
    }
    if (::find(m_options, name) != m_options.end())
    {
        throw std::runtime_error(scat("cmdline: duplicated option [", name, "]"));
    }

    if (!short_name.empty())
    {
        if (short_name.size() != 1 || short_name[0] == '-')
        {
            throw std::runtime_error(scat("cmdline: invalid short option name [", short_name, "]"));
        }
        if (::find(m_options, short_name) != m_options.end())
        {
            throw std::runtime_error(scat("cmdline: duplicated short option [", short_name, "]"));
        }
    }

    m_options.emplace_back(std::move(short_name), std::move(name), std::move(description), std::move(default_value));
}

cmdline_t::result_t cmdline_t::process(const int argc, const char* argv[]) const
{
    std::vector<std::pair<string_t, int>> tokens;
    for (int i = 1; i < argc; ++i)
    {
        string_t token = argv[i];
        assert(!token.empty());

        // NOLINTNEXTLINE(readability-suspicious-call-argument)
        if (nano::starts_with(token, str_dash_dash))
        {
            if (token.size() == 2U)
            {
                throw std::runtime_error(scat("cmdline: invalid option name [", token, "]"));
            }

            tokens.emplace_back(token.substr(2), 0);
        }
        // NOLINTNEXTLINE(readability-suspicious-call-argument)
        else if (nano::starts_with(token, str_dash))
        {
            if (token.size() != 2U)
            {
                throw std::runtime_error(scat("cmdline: invalid short option name [", token, "]"));
            }

            tokens.emplace_back(token.substr(1), 1);
        }
        else
        {
            tokens.emplace_back(std::move(token), 2);
        }
    }

    const auto get_mandatory_value = [&](const string_t& name, size_t i)
    {
        if (i + 1 == tokens.size())
        {
            throw std::runtime_error(scat("cmdline: expecting a value for option [", name, "]"));
        }

        const auto& [value, type] = tokens[i + 1];
        if (type != 2)
        {
            throw std::runtime_error(scat("cmdline: expecting a value for option [", name, "]"));
        }

        return value;
    };

    result_t result;
    for (const auto& option : m_options)
    {
        if (!option.m_default_value.empty())
        {
            result.m_ovalues[option.m_name] = option.m_default_value;
        }
    }

    for (size_t i = 0; i < tokens.size();)
    {
        const auto& [name, type] = tokens[i];

        if (type == 2)
        {
            throw std::runtime_error(scat("cmdline: value [", name, "] should follow a long or short option"));
        }

        const auto it = ::find(m_options, name);
        if (it == m_options.end())
        {
            result.m_xvalues[name] = get_mandatory_value(name, i);
            i += 2;
        }
        else
        {
            if (it->m_default_value.empty())
            {
                result.m_ovalues[it->m_name] = string_t{};
                i += 1;
            }
            else
            {
                result.m_ovalues[it->m_name] = get_mandatory_value(name, i);
                i += 2;
            }
        }
    }

    return result;
}

cmdline_t::result_t cmdline_t::process(const string_t& config) const
{
    strings_t tokens;
    for (auto tokenizer = tokenizer_t{config, " \t\n\r"}; tokenizer; ++tokenizer)
    {
        tokens.emplace_back(tokenizer.get());
    }

    std::vector<const char*> ptokens(tokens.size() + 1, nullptr);
    std::transform(tokens.begin(), tokens.end(), ptokens.begin() + 1, [&](const auto& token) { return token.c_str(); });

    return process(static_cast<int>(ptokens.size()), ptokens.data());
}

cmdline_t::result_t cmdline_t::process_config_file(const string_t& path) const
{
    std::ifstream  in(path.c_str());
    const string_t config((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    return process(config);
}

void cmdline_t::usage(std::ostream& os, size_t indent) const
{
    os << m_title << std::endl;

    size_t max_option_size = 0;
    for (const auto& option : m_options)
    {
        max_option_size = std::max(max_option_size, option.describe().size());
    }

    max_option_size += 4;
    for (const auto& option : m_options)
    {
        os << string_t(indent, ' ') << nano::align(option.describe(), max_option_size) << option.m_description
           << std::endl;
    }

    os << std::endl;
}
