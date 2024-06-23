#pragma once

#include <nano/logger.h>

namespace nano
{
///
/// \brief interface for objects that can optionally have a logger attached.
///
class NANO_PUBLIC loggable_t
{
public:
    ///
    /// \brief constructor
    ///
    loggable_t() = default;

    ///
    /// \brief attach a new logger.
    ///
    void logger(logger_t logger) { m_logger = std::move(logger); }

    ///
    /// \brief return the currently attached logger.
    ///
    const logger_t& logger() const { return m_logger; }

    ///
    /// \brief log the given tokens.
    ///
    template <class... targs>
    const logger_t& log(const targs&... args) const
    {
        return m_logger.log(args...);
    }

    ///
    /// \brief log the given tokens using the information level.
    ///
    template <class... targs>
    const logger_t& log_info(const targs&... args)
    {
        return log(log_type::info, args...);
    }

    ///
    /// \brief log the given tokens using the warning level.
    ///
    template <class... targs>
    const logger_t& log_warn(const targs&... args)
    {
        return log(log_type::warn, args...);
    }

    ///
    /// \brief log the given tokens using the error level.
    ///
    template <class... targs>
    const logger_t& log_error(const targs&... args)
    {
        return log(log_type::error, args...);
    }

private:
    // attributes
    logger_t m_logger; ///<
};
} // namespace nano
