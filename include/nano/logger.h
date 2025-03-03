#pragma once

#include <cstdint>
#include <memory>
#include <nano/arch.h>
#include <nano/string.h>
#include <ostream>

namespace nano
{
///
/// \brief stream header indicating a message with various severity levels of the format:
///     `[yyyy-mm-dd|hh:mm:ss]` and an appropriate color scheme.
///
enum class log_type : uint8_t
{
    info,  ///< information (e.g. green)
    warn,  ///< warning (e.g. orange)
    error, ///< error detected (e.g. red)
};

NANO_PUBLIC std::ostream& operator<<(std::ostream&, log_type);

///
/// \brief logging utility.
///
class NANO_PUBLIC logger_t
{
public:
    ///
    /// \brief default constructor (no logging).
    ///
    logger_t();

    ///
    /// \brief constructor (log to the given stream).
    ///
    explicit logger_t(std::ostream&);

    ///
    /// \brief constructor (log to the given file path).
    ///
    /// NB: the parent directories are created recursively if needed.
    ///
    explicit logger_t(string_t path);

    ///
    /// \brief enable copying.
    ///
    logger_t(const logger_t&);
    logger_t& operator=(const logger_t&);

    ///
    /// \brief enable moving.
    ///
    logger_t(logger_t&&) noexcept;
    logger_t& operator=(logger_t&&) noexcept;

    ///
    /// \brief destructor
    ///
    ~logger_t();

    ///
    /// \brief return the current logging prefix.
    ///
    const string_t& prefix() const;

    ///
    /// \brief set a prefix to use for all information, warning and errors logging level calls.
    ///
    const logger_t& prefix(string_t prefix) const;

    ///
    /// \brief create a logger to the file path: `current_parent_directory` / `filename`.
    ///
    logger_t fork(const string_t& filename) const;

    ///
    /// \brief create a logger to the file path: `current_parent_directory` / `directory / `filename`.
    ///
    logger_t fork(const string_t& directory, const string_t& filename) const;

    ///
    /// \brief log the given tokens.
    /// FIXME: more efficient logging so that the arguments are evaluated only if the logger is active.
    ///
    template <class... targs>
    const logger_t& log(const targs&... args) const
    {
        if (std::ostream* stream = this->stream(); stream != nullptr)
        {
            ((*stream) << ... << args);
        }
        return *this;
    }

    ///
    /// \brief log the given tokens using the information level.
    ///
    template <class... targs>
    const logger_t& info(const targs&... args) const
    {
        return log(log_type::info, prefix(), args...);
    }

    ///
    /// \brief log the given tokens using the warning level.
    ///
    template <class... targs>
    const logger_t& warn(const targs&... args) const
    {
        return log(log_type::warn, prefix(), args...);
    }

    ///
    /// \brief log the given tokens using the error level.
    ///
    template <class... targs>
    const logger_t& error(const targs&... args) const
    {
        return log(log_type::error, prefix(), args...);
    }

private:
    std::ostream* stream() const;

    // attributes
    class impl_t;
    std::unique_ptr<impl_t> m_pimpl;  ///< implementation details
};

///
/// \brief create a null logger.
///
NANO_PUBLIC logger_t make_null_logger();

///
/// \brief create a logger to the standard output streams.
///
NANO_PUBLIC logger_t make_stdout_logger();
NANO_PUBLIC logger_t make_stderr_logger();

///
/// \brief create a logger to the given stream.
///
NANO_PUBLIC logger_t make_stream_logger(std::ostream&);

///
/// \brief create a logger to the given file path.
///
/// NB: the parent directories are created recursively if needed.
///
NANO_PUBLIC logger_t make_file_logger(string_t path);

///
/// \brief RAII utility to append a particular logging prefix in the current scope.
///
class logger_prefix_scope_t
{
public:
    logger_prefix_scope_t(const logger_t& logger, string_t prefix)
        : m_logger(logger)
        , m_prefix(logger.prefix())
    {
        m_logger.prefix(m_prefix + std::move(prefix));
    }

    logger_prefix_scope_t(logger_prefix_scope_t&&)      = delete;
    logger_prefix_scope_t(const logger_prefix_scope_t&) = delete;

    logger_prefix_scope_t& operator=(logger_prefix_scope_t&&)      = delete;
    logger_prefix_scope_t& operator=(const logger_prefix_scope_t&) = delete;

    ~logger_prefix_scope_t() { m_logger.prefix(std::move(m_prefix)); }

private:
    const logger_t& m_logger;
    string_t        m_prefix;
};
} // namespace nano
