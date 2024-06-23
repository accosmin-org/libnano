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

private:
    std::ostream* stream() const;

    // attributes
    class impl_t;
    std::unique_ptr<impl_t> m_pimpl; ///< implementation details
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
} // namespace nano
