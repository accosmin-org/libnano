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
        explicit logger_t(const type, std::ostream* cout = nullptr, std::ostream* cerr = nullptr);

        ///
        /// \brief destructor
        ///
        ~logger_t();

        ///
        /// \brief log tokens
        ///
        template <typename T>
        logger_t& operator<<(const T& data)
        {
            m_stream << data;
            return *this;
        }

        ///
        /// \brief log manipulators
        ///
        logger_t& operator<<(std::ostream& (*pf)(std::ostream&))
        {
            pf(m_stream);
            return *this;
        }

    private:

        // attributes
        std::ostream&   m_stream;       ///< stream to write into
        std::streamsize m_precision;    ///< original precision to restore
    };

    ///
    /// \brief specific [information, warning, error] line loggers.
    ///
    inline logger_t log_info(std::ostream* cout = nullptr, std::ostream* cerr = nullptr)
    {
        return logger_t(logger_t::type::info, cout, cerr);
    }

    inline logger_t log_warning(std::ostream* cout = nullptr, std::ostream* cerr = nullptr)
    {
        return logger_t(logger_t::type::warn, cout, cerr);
    }

    inline logger_t log_error(std::ostream* cout = nullptr, std::ostream* cerr = nullptr)
    {
        return logger_t(logger_t::type::error, cout, cerr);
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
