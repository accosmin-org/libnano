#include <cassert>
#include <ctime>
#include <iomanip>
#include <nano/core/logger.h>

using namespace nano;

namespace
{
std::ostream*& info_stream()
{
    static auto* stream = &std::cout; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    return stream;
}

std::ostream*& warn_stream()
{
    static auto* stream = &std::cout; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    return stream;
}

std::ostream*& error_stream()
{
    static auto* stream = &std::cerr; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
    return stream;
}

const char* get_header(const logger_t::type type)
{
    switch (type)
    {
    case logger_t::type::info: return "\033[32m";
    case logger_t::type::warn: return "\033[33m";
    default: return "\033[31m";
    }
}
} // namespace

std::ostream& logger_t::stream(const logger_t::type type)
{
    switch (type)
    {
    case logger_t::type::info: return *::info_stream();
    case logger_t::type::warn: return *::warn_stream();
    default: return *::error_stream();
    }
}

std::ostream& logger_t::stream(const logger_t::type type, std::ostream& stream)
{
    switch (type)
    {
    case logger_t::type::info: ::info_stream() = &stream; break;
    case logger_t::type::warn: ::warn_stream() = &stream; break;
    case logger_t::type::error: ::error_stream() = &stream; break;
    default: assert(false);
    }
    return stream;
}

logger_t::logger_t(const type ltype)
    : m_stream(logger_t::stream(ltype))
    , m_precision(m_stream.precision())
{
    const std::time_t t = std::time(nullptr);

    // FIXME: Use the portable thread safe version in C++20!
    std::tm buff{};
#ifdef _WIN32
    ::localtime_s(&buff, &t);
#else // POSIX
    ::localtime_r(&t, &buff);
#endif
    m_stream << get_header(ltype) << "[" << std::put_time(&buff, "%F|%T") << "]\033[0m: ";
    m_stream << std::fixed << std::setprecision(6);
}

logger_t::~logger_t()
{
    m_stream << std::endl;
    m_stream.precision(m_precision);
}

logger_section_t::logger_section_t(std::ostream& info_stream, std::ostream& warn_stream, std::ostream& error_stream)
    : m_stream_info(&logger_t::stream(logger_t::type::info))
    , m_stream_warn(&logger_t::stream(logger_t::type::warn))
    , m_stream_error(&logger_t::stream(logger_t::type::error))
{
    logger_t::stream(logger_t::type::info, info_stream);
    logger_t::stream(logger_t::type::warn, warn_stream);
    logger_t::stream(logger_t::type::error, error_stream);
}

logger_section_t::~logger_section_t()
{
    logger_t::stream(logger_t::type::info, *m_stream_info);
    logger_t::stream(logger_t::type::warn, *m_stream_warn);
    logger_t::stream(logger_t::type::error, *m_stream_error);
}
