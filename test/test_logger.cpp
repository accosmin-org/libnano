#include <sstream>
#include <utest/utest.h>
#include <nano/logger.h>
#include <nano/string_utils.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_logger)

UTEST_CASE(info)
{
    std::ostringstream stream_cout, stream_cerr;
    log_info(true, &stream_cout, &stream_cerr) << "info message";

    UTEST_CHECK(ends_with(stream_cout.str(), "|i]: info message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_CASE(error)
{
    std::ostringstream stream_cout, stream_cerr;
    log_error(false, &stream_cout, &stream_cerr) << "error message";

    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK(ends_with(stream_cerr.str(), "|e]: error message\n"));
}

UTEST_CASE(warning)
{
    std::ostringstream stream_cout, stream_cerr;
    log_warning(false, &stream_cout, &stream_cerr) << "warning message";

    UTEST_CHECK(ends_with(stream_cout.str(), "|w]: warning message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_END_MODULE()
