#include <sstream>
#include <iomanip>
#include <utest/utest.h>
#include <nano/logger.h>
#include <nano/string.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_logger)

UTEST_CASE(log)
{
    UTEST_CHECK_NOTHROW(log_info() << "info message");
    UTEST_CHECK_NOTHROW(log_error() << "error message");
    UTEST_CHECK_NOTHROW(log_warning() << "warning message");
    UTEST_CHECK_NOTHROW(logger_t(static_cast<logger_t::type>(42)) << "what message");
}

UTEST_CASE(info)
{
    std::ostringstream stream_cout, stream_cerr;
    log_info(&stream_cout, &stream_cerr) << std::flush << "info message" << '\n' << std::endl;

    UTEST_CHECK(ends_with(stream_cout.str(), ": info message\n\n\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_CASE(error)
{
    std::ostringstream stream_cout, stream_cerr;
    stream_cerr << std::setprecision(3);
    log_error(&stream_cout, &stream_cerr) << std::setprecision(7) << "error message";

    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK(ends_with(stream_cerr.str(), ": error message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.precision(), 3);
}

UTEST_CASE(warning)
{
    std::ostringstream stream_cout, stream_cerr;
    log_warning(&stream_cout, &stream_cerr) << "warning message";

    UTEST_CHECK(ends_with(stream_cout.str(), ": warning message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_CASE(unknown)
{
    std::ostringstream stream_cout, stream_cerr;
    logger_t(static_cast<logger_t::type>(42), &stream_cout, &stream_cerr) << "unknown message";

    UTEST_CHECK(ends_with(stream_cout.str(), ": unknown message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_CASE(critical)
{
    UTEST_CHECK_NOTHROW(critical(false, "message"));
    UTEST_CHECK_THROW(critical(true, "message"), std::runtime_error);
}

UTEST_END_MODULE()
