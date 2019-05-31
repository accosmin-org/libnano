#include <fstream>
#include <utest/utest.h>
#include <nano/cmdline.h>

UTEST_BEGIN_MODULE(test_core_cmdline)

UTEST_CASE(parse)
{
    nano::cmdline_t cmdline("unit testing");
    cmdline.add("v", "version", "version", "0.3");
    cmdline.add("", "iterations", "number of iterations", "127");

    const int argc = 4;
    const char* argv[] = { "", "-v", "--iterations", "7" };

    cmdline.process(argc, argv);

    UTEST_CHECK(cmdline.has("v"));
    UTEST_CHECK(cmdline.has("version"));
    UTEST_CHECK(cmdline.has("iterations"));
    UTEST_CHECK(!cmdline.has("h"));
    UTEST_CHECK(!cmdline.has("help"));

    UTEST_CHECK_EQUAL(cmdline.get<int>("iterations"), 7);
    UTEST_CHECK_EQUAL(cmdline.get<std::string>("v"), "0.3");
}

UTEST_CASE(empty)
{
    nano::cmdline_t cmdline("unit testing");

    UTEST_CHECK(!cmdline.has("h"));
    UTEST_CHECK(!cmdline.has("help"));
    UTEST_CHECK_THROW(cmdline.has("v"), std::runtime_error);

    UTEST_CHECK_THROW(cmdline.get<int>("version"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.get<std::string>("f"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.get<std::string>("file"), std::runtime_error);
}

UTEST_CASE(error_invalid_arg)
{
    nano::cmdline_t cmdline("unit testing");
    cmdline.add("v", "version", "version");
    cmdline.add("", "iterations", "number of iterations", "127");

    const int argc = 4;
    const char* argv[] = { "", "v", "--version", "7" };

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_unknown_arg)
{
    nano::cmdline_t cmdline("unit testing");
    cmdline.add("v", "version", "version");
    cmdline.add("", "iterations", "number of iterations", "127");

    const int argc = 4;
    const char* argv[] = { "", "-v", "--what", "7" };

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(parse_config_file)
{
    nano::cmdline_t cmdline("unit testing");
    cmdline.add("v", "version", "version", "0.3");
    cmdline.add("", "iterations", "number of iterations", "127");

    const std::string path = "config";

    {
        std::ofstream out(path.c_str());
        out << "-v\n";
        out << "--iterations 29";
    }

    cmdline.process_config_file(path);

    UTEST_CHECK(cmdline.has("v"));
    UTEST_CHECK(cmdline.has("version"));
    UTEST_CHECK(cmdline.has("iterations"));
    UTEST_CHECK(!cmdline.has("h"));
    UTEST_CHECK(!cmdline.has("help"));

    UTEST_CHECK_EQUAL(cmdline.get<std::string>("v"), "0.3");
    UTEST_CHECK_EQUAL(cmdline.get<int>("iterations"), 29);

    std::remove(path.c_str());
}

UTEST_END_MODULE()
