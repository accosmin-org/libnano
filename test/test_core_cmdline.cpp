#include <filesystem>
#include <fstream>
#include <nano/core/cmdline.h>
#include <utest/utest.h>

static void check(const nano::cmdline_t::result_t& result, const nano::cmdline_t::result_t::storage_t& expected_ovalues,
                  const nano::cmdline_t::result_t::storage_t& expected_xvalues)
{
    UTEST_CHECK_EQUAL(result.m_ovalues.size(), expected_ovalues.size());
    UTEST_CHECK_EQUAL(result.m_xvalues.size(), expected_xvalues.size());

    for (const auto& [k, v] : expected_ovalues)
    {
        UTEST_CHECK(result.has(k));
        if (v.empty())
        {
            const auto& kk = k;
            UTEST_CHECK_THROW(result.get<std::string>(kk), std::runtime_error);
        }
        else
        {
            UTEST_CHECK_EQUAL(result.get<std::string>(k), v);

            if (v == "42")
            {
                UTEST_CHECK_EQUAL(result.get<int>(k), 42);
            }
            else if (v == "xy")
            {
                const auto& kk = k;
                UTEST_CHECK_THROW(result.get<int>(kk), std::invalid_argument);
            }
        }
    }

    for (const auto& [k, v] : expected_xvalues)
    {
        const auto it = result.m_xvalues.find(k);
        UTEST_REQUIRE(it != result.m_xvalues.end());
        UTEST_CHECK_EQUAL(it->second, v);
    }

    UTEST_CHECK(!result.has("what?!"));
    UTEST_CHECK_THROW(result.get<int>("what?!"), std::runtime_error);
    UTEST_CHECK_THROW(result.get<std::string>("what?!"), std::runtime_error);
}

UTEST_BEGIN_MODULE(test_core_cmdline)

UTEST_CASE(empty)
{
    const auto cmdline = nano::cmdline_t{"unit testing"};

    std::stringstream os;
    cmdline.usage(os);

    UTEST_CHECK_EQUAL(os.str(), R"XXX(unit testing
  -h,--help    usage

)XXX");
}

UTEST_CASE(usage)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("d", "doit", "do something important if set"));
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version number", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", 100));

    std::stringstream os;
    cmdline.usage(os);

    UTEST_CHECK_EQUAL(os.str(), R"XXX(unit testing
  -h,--help            usage
  -d,--doit            do something important if set
  -v,--version(0.3)    version number
  --iterations(100)    number of iterations

)XXX");
}

UTEST_CASE(parse_chars)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "trials", "number of trials"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations"));

    const int   argc   = 3;
    const char* argv[] = {"", "-v", "0.3.1"};

    check(cmdline.process(argc, argv),
          {
              {"version", "0.3.1"},
    },
          {});
}

UTEST_CASE(parse_string)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("", "doit", "do something important if set"));
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", 127));

    check(cmdline.process("--help --iterations 7"),
          {
              {      "help",    ""},
              {   "version", "0.3"},
              {"iterations",   "7"},
    },
          {});

    check(cmdline.process("-v 1.0 --extra1 value1 --extra2 value2 -x value3"),
          {
              {   "version", "1.0"},
              {"iterations", "127"},
    },
          {
              {"extra1", "value1"},
              {"extra2", "value2"},
              {"x", "value3"},
          });
}

UTEST_CASE(error_invalid_options)
{
    nano::cmdline_t cmdline("unit testing");

    UTEST_CHECK_THROW(cmdline.add("v", "", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("v", "-", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("v", "--", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("v", "--version", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-", "version", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("vv", "version", ""), std::runtime_error);
}

UTEST_CASE(error_duplicate_options)
{
    nano::cmdline_t cmdline("unit testing");

    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", ""));
    UTEST_CHECK_THROW(cmdline.add("v", "version", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("v", "wersion", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("w", "version", ""), std::runtime_error);
}

UTEST_CASE(error_invalid_arg1)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "v", "--version", "7"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_invalid_arg2)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "--version", "7", "-", "--version", "13"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_invalid_arg3)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "--version", "11", "--"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_value_without_option)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 5;
    const char* argv[] = {"", "-v", "--extra", "7", "17"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_last_value_without_option)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 5;
    const char* argv[] = {"", "-v", "--extra", "7", "--another-extra"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_option_with_default_and_no_value)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 5;
    const char* argv[] = {"", "-v", "--iterations", "--extra", "7"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(parse_config_file)
{
    nano::cmdline_t cmdline("unit testing");
    UTEST_CHECK_NOTHROW(cmdline.add("v", "version", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const std::string path = std::filesystem::temp_directory_path() / "libnano.config.tmp";
    {
        std::ofstream out(path.c_str());
        out << "-v\n";
        out << "--iterations xy\n";
        out << "--extra 42\n";
    }

    check(cmdline.process_config_file(path),
          {
              {   "version",   ""},
              {"iterations", "xy"}
    },
          {{"extra", "42"}});
}

UTEST_END_MODULE()
