#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nano/core/overloaded.h>
#include <nano/logger.h>
#include <variant>

using namespace nano;

namespace
{
constexpr const char* TERMINAL_RESET_COLOR  = "\033[0m";
constexpr const char* TERMINAL_COLOR_GREEN  = "\033[32m";
constexpr const char* TERMINAL_COLOR_YELLOW = "\033[33m";
constexpr const char* TERMINAL_COLOR_RED    = "\033[31m";

const char* header_to_color(const log_type type)
{
    switch (type)
    {
    case log_type::info:
        return TERMINAL_COLOR_GREEN;
    case log_type::warn:
        return TERMINAL_COLOR_YELLOW;
    default:
        return TERMINAL_COLOR_RED;
    }
}

auto make_path(std::filesystem::path path)
{
    if (!path.empty() && !path.parent_path().empty())
    {
        std::filesystem::create_directories(path.parent_path());
    }
    return path;
}
} // namespace

std::ostream& nano::operator<<(std::ostream& stream, const log_type type)
{
    const auto time = std::time(nullptr);

    // FIXME: Use the portable thread safe version in C++20!
    std::tm buff{};
#ifdef _WIN32
    ::localtime_s(&buff, &time);
#else // POSIX
    [[maybe_unused]] const auto* const _ = ::localtime_r(&time, &buff);
#endif
    return stream << header_to_color(type) << '[' << std::put_time(&buff, "%F|%T") << ']' << TERMINAL_RESET_COLOR
                  << ' ';
}

class logger_t::impl_t
{
public:
    impl_t() = default;

    explicit impl_t(std::ostream& stream, string_t prefix = string_t{})
        : m_storage(&stream)
        , m_prefix(std::move(prefix))
    {
    }

    explicit impl_t(string_t path, string_t prefix = string_t{})
        : m_path(make_path(std::move(path)))
        , m_storage(std::ofstream{m_path})
        , m_prefix(std::move(prefix))
    {
    }

    explicit impl_t(std::ostream* stream, string_t path, string_t prefix = string_t{})
        : m_path(std::move(path))
        , m_prefix(std::move(prefix))
    {
        if (stream != nullptr)
        {
            m_storage = stream;
        }
    }

    const string_t& prefix() const { return m_prefix; }

    const std::filesystem::path& path() const { return m_path; }

    std::filesystem::path parent_path() const { return m_path.parent_path(); }

    std::ostream* stream()
    {
        return std::visit(overloaded{[&](std::monostate&) -> std::ostream* { return nullptr; },
                                     [&](std::ostream* stream) -> std::ostream* { return stream; },
                                     [&](std::ofstream& stream) -> std::ostream* { return &stream; }},
                          m_storage);
    }

    void prefix(string_t prefix) { m_prefix = std::move(prefix); }

private:
    using storage_t = std::variant<std::monostate, std::ostream*, std::ofstream>;

    // attributes
    std::filesystem::path m_path;
    storage_t             m_storage;
    string_t              m_prefix;
};

logger_t::logger_t()
    : m_pimpl(std::make_unique<impl_t>())
{
}

logger_t::logger_t(std::ostream& stream)
    : m_pimpl(std::make_unique<impl_t>(stream))
{
}

logger_t::logger_t(string_t path)
    : m_pimpl(std::make_unique<impl_t>(std::move(path)))
{
}

logger_t::logger_t(logger_t&&) noexcept = default;

logger_t::logger_t(const logger_t& other)
    : m_pimpl(std::make_unique<impl_t>(other.stream(), other.m_pimpl->path().string(), other.m_pimpl->prefix()))
{
}

logger_t& logger_t::operator=(logger_t&&) noexcept = default;

logger_t& logger_t::operator=(const logger_t& other)
{
    if (this != &other)
    {
        m_pimpl = std::make_unique<impl_t>(other.stream(), other.m_pimpl->path().string(), other.m_pimpl->prefix());
    }
    return *this;
}

logger_t::~logger_t() = default;

const string_t& logger_t::prefix() const
{
    return m_pimpl->prefix();
}

const logger_t& logger_t::prefix(string_t prefix) const
{
    m_pimpl->prefix(std::move(prefix));
    return *this;
}

logger_t logger_t::fork(const string_t& filename) const
{
    return make_file_logger((m_pimpl->parent_path() / filename).string());
}

logger_t logger_t::fork(const string_t& directory, const string_t& filename) const
{
    return make_file_logger((m_pimpl->parent_path() / directory / filename).string());
}

std::ostream* logger_t::stream() const
{
    return m_pimpl->stream();
}

logger_t nano::make_null_logger()
{
    return {};
}

logger_t nano::make_stdout_logger()
{
    return make_stream_logger(std::cout);
}

logger_t nano::make_stderr_logger()
{
    return make_stream_logger(std::cerr);
}

logger_t nano::make_stream_logger(std::ostream& stream)
{
    return logger_t{stream};
}

logger_t nano::make_file_logger(string_t path)
{
    return logger_t{std::move(path)};
}
