#pragma once

#include <ostream>
#include <nano/arch.h>
#include <nano/chrono.h>

namespace nano
{
    ///
    /// \brief logging object.
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
        logger_t(const type, const bool flush_at_endl = true);

        ///
        /// \brief destructor
        ///
        ~logger_t();

        ///
        /// \brief log element
        ///
        template <typename T>
        logger_t& operator<<(const T& data)
        {
            m_stream << data;
            return *this;
        }
        logger_t& operator<<(std::ostream& (*pf)(std::ostream&));
        logger_t& operator<<(logger_t& (*pf)(logger_t&));

        ///
        /// \brief log tags
        ///
        logger_t& newl();
        logger_t& endl();
        logger_t& flush();

    private:

        // attributes
        std::ostream&   m_stream;       ///< stream to write into
        std::streamsize m_precision;    ///< original precision to restore
        bool            m_flush{true};
    };

    ///
    /// \brief stream particular tags
    ///
    inline logger_t& newl(logger_t& logger)     { return logger.newl(); }
    inline logger_t& endl(logger_t& logger)     { return logger.endl(); }
    inline logger_t& flush(logger_t& logger)    { return logger.flush(); }

    ///
    /// \brief specific [information, warning, error] line loggers.
    ///
    inline logger_t log_info(const bool flush_at_destruction = true)
    {
        return logger_t(logger_t::type::info, flush_at_destruction);
    }
    inline logger_t log_warning(const bool flush_at_destruction = true)
    {
        return logger_t(logger_t::type::warn, flush_at_destruction);
    }
    inline logger_t log_error(const bool flush_at_destruction = true)
    {
        return logger_t(logger_t::type::error, flush_at_destruction);
    }

    ///
    /// \brief run and check a critical step (checkpoint).
    ///
    template <typename tresult, typename tstring>
    void critical(const tresult& result, const tstring& message)
    {
        const timer_t timer;
        if (static_cast<bool>(result))
        {
            log_info() << message << " done in [" << timer.elapsed() << "].";
        }
        else
        {
            log_error() << message << " failed after [" << timer.elapsed() << "]!";
            throw std::runtime_error("critical check failed");
        }
    }

    ///
    /// \brief wraps main function to catch and log all exceptions.
    ///
    template <typename toperator>
    int main(const toperator& op, int argc, const char* argv[])
    {
        try
        {
            return op(argc, argv);
        }
        catch (std::exception& e)
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
}
