#pragma once

#include <ctime>
#include <string>
#include <iomanip>
#include <iostream>

namespace nano
{
    ///
    /// \brief logging object.
    ///
    class logger_t
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
        explicit logger_t(const type ltype, std::ostream* cout = &std::cout, std::ostream* cerr = &std::cerr) :
            m_stream(get_stream(ltype, cout, cerr)),
            m_precision(m_stream.precision())
        {
            const std::time_t t = std::time(nullptr);
            m_stream << "[" << std::put_time(std::localtime(&t), "%F|%T") << "|" << get_header(ltype) << "]: ";
            m_stream << std::fixed << std::setprecision(6);
        }

        ///
        /// \brief disable copying
        ///
        logger_t(const logger_t&) = delete;
        logger_t& operator=(const logger_t&) = delete;

        ///
        /// \brief disable moving
        ///
        logger_t(logger_t&&) noexcept = default;
        logger_t& operator=(logger_t&&) noexcept = delete;

        ///
        /// \brief destructor
        ///
        ~logger_t()
        {
            m_stream << std::endl;
            m_stream.precision(m_precision);
        }

        ///
        /// \brief log tokens
        ///
        template <typename T>
        const logger_t& operator<<(const T& data) const
        {
            m_stream << data;
            return *this;
        }

        ///
        /// \brief log manipulators
        ///
        const logger_t& operator<<(std::ostream& (*pf)(std::ostream&)) const
        {
            pf(m_stream);
            return *this;
        }

    private:

        static std::ostream& get_stream(const logger_t::type type, std::ostream* cout, std::ostream* cerr)
        {
            switch (type)
            {
            case logger_t::type::info:      return (cout != nullptr ? *cout : std::cout);
            case logger_t::type::warn:      return (cout != nullptr ? *cout : std::cout);
            case logger_t::type::error:     return (cerr != nullptr ? *cerr : std::cerr);
            default:                        return (cout != nullptr ? *cout : std::cout);
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

        // attributes
        std::ostream&   m_stream;       ///< stream to write into
        std::streamsize m_precision;    ///< original precision to restore
    };

    ///
    /// \brief specific [information, warning, error] line loggers.
    ///
    inline logger_t log_info(std::ostream* cout = &std::cout, std::ostream* cerr = &std::cerr)
    {
        return logger_t(logger_t::type::info, cout, cerr);
    }

    inline logger_t log_warning(std::ostream* cout = &std::cout, std::ostream* cerr = &std::cerr)
    {
        return logger_t(logger_t::type::warn, cout, cerr);
    }

    inline logger_t log_error(std::ostream* cout = &std::cout, std::ostream* cerr = &std::cerr)
    {
        return logger_t(logger_t::type::error, cout, cerr);
    }

    ///
    /// \brief checks and throws an exception if the given condition is satisfied.
    ///
    template <typename tresult, typename tstring>
    void critical(const tresult& result, const tstring& message)
    {
        if (static_cast<bool>(result))
        {
            log_error() << message;
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
