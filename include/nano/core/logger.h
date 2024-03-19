#pragma once

#include <iostream>
#include <nano/arch.h>
#include <nano/string.h>

namespace nano
{
///
/// \brief logging object that uses a global streaming device.
///
class NANO_PUBLIC logger_t
{
public:
    enum class type
    {
        info,
        warn,
        error
    };

    ///
    /// \brief constructor
    ///
    explicit logger_t(type);

    ///
    /// \brief disable copying
    ///
    logger_t(const logger_t&)            = delete;
    logger_t& operator=(const logger_t&) = delete;

    ///
    /// \brief disable moving
    ///
    logger_t(logger_t&&) noexcept            = default;
    logger_t& operator=(logger_t&&) noexcept = delete;

    ///
    /// \brief destructor
    ///
    ~logger_t();

    ///
    /// \brief log tokens.
    ///
    template <typename T>
    const logger_t& operator<<(const T& data) const
    {
        m_stream << data;
        return *this;
    }

    ///
    /// \brief log manipulators.
    ///
    const logger_t& operator<<(std::ostream& (*pf)(std::ostream&)) const
    {
        pf(m_stream);
        return *this;
    }

    ///
    /// \brief return the current streaming device associated to the given logging level.
    ///
    static std::ostream& stream(type);

    ///
    /// \brief change the current streaming device associated to the given logging level.
    ///
    static std::ostream& stream(type, std::ostream&);

private:
    // attributes
    std::ostream&   m_stream;    ///< stream to write into
    std::streamsize m_precision; ///< original precision to restore
};

///
/// \brief RAII utility to setup the given streaming devices during its lifetime.
///
class NANO_PUBLIC logger_section_t
{
public:
    ///
    /// \brief constructor
    ///
    logger_section_t(std::ostream& info_stream, std::ostream& warn_stream, std::ostream& error_stream);

    ///
    /// \brief disable copying and moving.
    ///
    logger_section_t(logger_section_t&&)                 = delete;
    logger_section_t(const logger_section_t&)            = delete;
    logger_section_t& operator=(logger_section_t&&)      = delete;
    logger_section_t& operator=(const logger_section_t&) = delete;

    ///
    /// \brief destructor (reset the global streaming devices).
    ///
    ~logger_section_t();

private:
    // attributes
    std::ostream* m_stream_info{nullptr};  ///<
    std::ostream* m_stream_warn{nullptr};  ///<
    std::ostream* m_stream_error{nullptr}; ///<
};

///
/// \brief specific [information, warning, error] line loggers using the current streaming device.
///
inline logger_t log_info()
{
    return logger_t(logger_t::type::info);
}

inline logger_t log_warning()
{
    return logger_t(logger_t::type::warn);
}

inline logger_t log_error()
{
    return logger_t(logger_t::type::error);
}

///
/// \brief throws an exception as a critical condition is satisfied.
/// FIXME: use std::source_location when moving to C++20 to automatically add this information
///
template <typename... tmessage>
[[noreturn]] void critical0(const tmessage&... message)
{
    (log_error() << ... << message);
    throw std::runtime_error("critical check failed!");
}

///
/// \brief checks and throws an exception if the given condition is satisfied.
/// FIXME: use std::source_location when moving to C++20 to automatically add this information
///
template <typename tcondition, typename... tmessage>
void critical(const tcondition& condition, const tmessage&... message)
{
    if (static_cast<bool>(condition))
    {
        critical0(message...);
    }
}

///
/// \brief wraps main function to catch and log all exceptions.
///
template <typename toperator>
int safe_main(const toperator& op, const int argc, const char* argv[])
{
    try
    {
        return op(argc, argv);
    }
    catch (const std::exception& e)
    {
        log_error() << "caught exception (" << e.what() << ")!";
        return EXIT_FAILURE;
    }
    catch (...)
    {
        log_error() << "caught unknown exception!";
        return EXIT_FAILURE;
    }
}
} // namespace nano
