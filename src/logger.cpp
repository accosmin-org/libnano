#include <ctime>
#include <string>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <nano/logger.h>

using namespace nano;

static std::ostream& get_stream(const logger_t::type type, std::ostream* cout, std::ostream* cerr)
{
    switch (type)
    {
    case logger_t::type::info:      return (cout ? *cout : std::cout);
    case logger_t::type::warn:      return (cout ? *cout : std::cout);
    case logger_t::type::error:     return (cerr ? *cerr : std::cerr);
    default:                        return (cout ? *cout : std::cout);
    }
}

static char get_header(const logger_t::type type)
{
    switch (type)
    {
    case logger_t::type::info:      return 'i';
    case logger_t::type::warn:      return 'w';
    case logger_t::type::error:     return 'e';
    default:                        return '?';
    }
}

logger_t::logger_t(const logger_t::type ltype, std::ostream* cout, std::ostream* cerr) :
    m_stream(get_stream(ltype, cout, cerr)),
    m_precision(m_stream.precision())
{
    const std::time_t t = std::time(nullptr);
    m_stream << "[" << std::put_time(std::localtime(&t), "%F|%T") << "|" << get_header(ltype) << "]: ";
    m_stream << std::fixed << std::setprecision(6);
}

logger_t::~logger_t()
{
    m_stream << std::endl;
    m_stream.precision(m_precision);
}
