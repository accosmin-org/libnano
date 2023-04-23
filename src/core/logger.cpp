#include <ctime>
#include <iomanip>
#include <nano/core/logger.h>
#include <string>

using namespace nano;

namespace
{
std::ostream& get_stream(logger_t::type type, std::ostream* cout, std::ostream* cerr)
{
    switch (type)
    {
    case logger_t::type::info:
    case logger_t::type::warn: return (cout != nullptr ? *cout : std::cout);
    case logger_t::type::error: return (cerr != nullptr ? *cerr : std::cerr);
    default: return (cout != nullptr ? *cout : std::cout);
    }
}

const char* get_header(logger_t::type type)
{
    switch (type)
    {
    case logger_t::type::info: return "\033[32m";
    case logger_t::type::warn: return "\033[33m";
    case logger_t::type::error: return "\033[31m";
    default: return "\033[91m";
    }
}
} // namespace

logger_t::logger_t(type ltype, std::ostream* cout, std::ostream* cerr)
    : m_stream(get_stream(ltype, cout, cerr))
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
