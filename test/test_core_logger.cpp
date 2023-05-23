#include <iomanip>
#include <nano/core/logger.h>
#include <nano/core/strutil.h>
#include <sstream>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_core_logger)

UTEST_CASE(_default)
{
    UTEST_CHECK(&logger_t::stream(logger_t::type::info) == &std::cout);
    UTEST_CHECK(&logger_t::stream(logger_t::type::warn) == &std::cout);
    UTEST_CHECK(&logger_t::stream(logger_t::type::error) == &std::cerr);

    UTEST_CHECK_NOTHROW(log_info() << "info message");
    UTEST_CHECK_NOTHROW(log_error() << "error message");
    UTEST_CHECK_NOTHROW(log_warning() << "warning message");
    UTEST_CHECK_NOTHROW(logger_t(static_cast<logger_t::type>(42)) << "what message");
}

UTEST_CASE(stream_section)
{
    {
        std::ostringstream stream_cout, stream_warn, stream_cerr;
        const auto         _ = logger_section_t{stream_cout, stream_warn, stream_cerr};

        UTEST_CHECK(&logger_t::stream(logger_t::type::info) == &stream_cout);
        UTEST_CHECK(&logger_t::stream(logger_t::type::warn) == &stream_warn);
        UTEST_CHECK(&logger_t::stream(logger_t::type::error) == &stream_cerr);

        log_info() << std::flush << "info message" << '\n' << std::endl;
        UTEST_CHECK(ends_with(stream_cout.str(), ": info message\n\n\n"));
        UTEST_CHECK_EQUAL(stream_warn.str(), "");
        UTEST_CHECK_EQUAL(stream_cerr.str(), "");

        stream_cout.str("");
        stream_warn.str("");
        stream_cerr.str("");

        stream_cerr << std::setprecision(3);
        log_error() << std::setprecision(7) << "error message";
        UTEST_CHECK_EQUAL(stream_cout.str(), "");
        UTEST_CHECK_EQUAL(stream_warn.str(), "");
        UTEST_CHECK(ends_with(stream_cerr.str(), ": error message\n"));
        UTEST_CHECK_EQUAL(stream_cerr.precision(), 3);

        stream_cout.str("");
        stream_warn.str("");
        stream_cerr.str("");

        log_warning() << "warning message";
        UTEST_CHECK_EQUAL(stream_cout.str(), "");
        UTEST_CHECK(ends_with(stream_warn.str(), ": warning message\n"));
        UTEST_CHECK_EQUAL(stream_cerr.str(), "");
    }
    UTEST_CHECK(&logger_t::stream(logger_t::type::info) == &std::cout);
    UTEST_CHECK(&logger_t::stream(logger_t::type::warn) == &std::cout);
    UTEST_CHECK(&logger_t::stream(logger_t::type::error) == &std::cerr);
}

UTEST_CASE(critical)
{
    UTEST_CHECK_NOTHROW(critical(false, "message"));
    UTEST_CHECK_THROW(critical(true, "message"), std::runtime_error);
}

UTEST_CASE(main)
{
    const auto op_ok = [](int, const char*[]) { return EXIT_SUCCESS; };

    const auto op_unknown = [](int, const char*[])
    {
        throw 42;            // NOLINT(hicpp-exception-baseclass)
        return EXIT_SUCCESS; // cppcheck-suppress duplicateBreak
    };

    const auto op_exception = [](int, const char*[])
    {
        throw std::runtime_error("runtime error");
        return EXIT_SUCCESS; // cppcheck-suppress duplicateBreak
    };

    const auto  argc   = 1;
    const char* argv[] = {"main"};

    UTEST_CHECK_NOTHROW(nano::safe_main(op_ok, argc, argv));
    UTEST_CHECK_NOTHROW(nano::safe_main(op_unknown, argc, argv));
    UTEST_CHECK_NOTHROW(nano::safe_main(op_exception, argc, argv));

    UTEST_CHECK_EQUAL(nano::safe_main(op_ok, argc, argv), EXIT_SUCCESS);
    UTEST_CHECK_EQUAL(nano::safe_main(op_unknown, argc, argv), EXIT_FAILURE);
    UTEST_CHECK_EQUAL(nano::safe_main(op_exception, argc, argv), EXIT_FAILURE);
}

UTEST_END_MODULE()
