#include <iomanip>
#include <iostream>
#include <utest/utest.h>
#include <nano/logger.h>
#include <nano/string_utils.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_logger)

UTEST_CASE(info)
{
    auto* old_rdbuf_cout = std::cout.rdbuf();
    auto* old_rdbuf_cerr = std::cerr.rdbuf();

    std::ostringstream stream_cout, stream_cerr;
    std::cout.rdbuf(stream_cout.rdbuf());
    std::cerr.rdbuf(stream_cerr.rdbuf());

    log_info() << "info message";

    std::cout.rdbuf(old_rdbuf_cout);
    std::cerr.rdbuf(old_rdbuf_cerr);

    UTEST_CHECK(ends_with(stream_cout.str(), "|i]: info message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_CASE(error)
{
    auto* old_rdbuf_cout = std::cout.rdbuf();
    auto* old_rdbuf_cerr = std::cerr.rdbuf();

    std::ostringstream stream_cout, stream_cerr;
    std::cout.rdbuf(stream_cout.rdbuf());
    std::cerr.rdbuf(stream_cerr.rdbuf());

    log_error() << "error message";

    std::cout.rdbuf(old_rdbuf_cout);
    std::cerr.rdbuf(old_rdbuf_cerr);

    UTEST_CHECK_EQUAL(stream_cout.str(), "");
    UTEST_CHECK(ends_with(stream_cerr.str(), "|e]: error message\n"));
}

UTEST_CASE(warning)
{
    auto* old_rdbuf_cout = std::cout.rdbuf();
    auto* old_rdbuf_cerr = std::cerr.rdbuf();

    std::ostringstream stream_cout, stream_cerr;
    std::cout.rdbuf(stream_cout.rdbuf());
    std::cerr.rdbuf(stream_cerr.rdbuf());

    log_warning() << "warning message";

    std::cout.rdbuf(old_rdbuf_cout);
    std::cerr.rdbuf(old_rdbuf_cerr);

    UTEST_CHECK(ends_with(stream_cout.str(), "|w]: warning message\n"));
    UTEST_CHECK_EQUAL(stream_cerr.str(), "");
}

UTEST_END_MODULE()
