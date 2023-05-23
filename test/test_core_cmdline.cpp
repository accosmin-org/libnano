#include <filesystem>
#include <fstream>
#include <nano/core/logger.h>
#include <nano/core/parameter_tracker.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
void check(const cmdline_t::result_t& result, const cmdline_t::result_t::storage_t& expected_ovalues,
           const cmdline_t::result_t::storage_t& expected_xvalues)
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
} // namespace

UTEST_BEGIN_MODULE(test_core_cmdline)

UTEST_CASE(empty)
{
    const auto cmdline = cmdline_t{"unit testing"};

    std::stringstream os;
    cmdline.usage(os);

    UTEST_CHECK_EQUAL(os.str(), R"XXX(unit testing
  -h,--help        usage
  -v,--version     library version
  -g,--git-hash    git commit hash

)XXX");
}

UTEST_CASE(usage)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("d", "doit", "do something important if set"));
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version number", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", 100));

    std::stringstream os;
    cmdline.usage(os);

    UTEST_CHECK_EQUAL(os.str(), R"XXX(unit testing
  -h,--help             usage
  -v,--version          library version
  -g,--git-hash         git commit hash
  -d,--doit             do something important if set
  -x,--xversion(0.3)    version number
  --iterations(100)     number of iterations

)XXX");
}

UTEST_CASE(parse_chars)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "trials", "number of trials"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations"));

    const int   argc   = 3;
    const char* argv[] = {"", "-x", "0.3.1"};

    check(cmdline.process(argc, argv),
          {
              {"xversion", "0.3.1"},
    },
          {});
}

UTEST_CASE(parse_string)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("", "doit", "do something important if set"));
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", 127));

    check(cmdline.process("--help --iterations 7"),
          {
              {      "help",    ""},
              {  "xversion", "0.3"},
              {"iterations",   "7"},
    },
          {});

    check(cmdline.process("-x 1.0 --extra1 value1 --extra2 value2 -y value3"),
          {
              {  "xversion", "1.0"},
              {"iterations", "127"},
    },
          {
              {"extra1", "value1"},
              {"extra2", "value2"},
              {"y", "value3"},
          });
}

UTEST_CASE(error_invalid_options)
{
    auto cmdline = cmdline_t{"unit testing"};

    UTEST_CHECK_THROW(cmdline.add("x", "", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("x", "-", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("x", "--", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("x", "--xversion", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-", "xversion", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("xx", "xversion", ""), std::runtime_error);
}

UTEST_CASE(error_duplicate_options)
{
    auto cmdline = cmdline_t{"unit testing"};

    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", ""));
    UTEST_CHECK_THROW(cmdline.add("x", "xversion", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("x", "xwersion", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("w", "xversion", ""), std::runtime_error);
}

UTEST_CASE(error_invalid_arg1)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "x", "--xversion", "7"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_invalid_arg2)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "--xversion", "7", "-", "--xversion", "13"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_invalid_arg3)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "--xversion", "11", "--"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_value_without_option)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 5;
    const char* argv[] = {"", "-x", "--extra", "7", "17"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_last_value_without_option)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 5;
    const char* argv[] = {"", "-x", "--extra", "7", "--another-extra"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(error_option_with_default_and_no_value)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const int   argc   = 5;
    const char* argv[] = {"", "-x", "--iterations", "--extra", "7"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(parse_config_file)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("x", "xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("", "iterations", "number of iterations", "127"));

    const auto path = (std::filesystem::temp_directory_path() / "libnano.config.tmp").string();
    {
        std::ofstream out(path.c_str());
        out << "-x\n";
        out << "--iterations xy\n";
        out << "--extra 42\n";
    }

    check(cmdline.process_config_file(path),
          {
              {  "xversion",   ""},
              {"iterations", "xy"}
    },
          {{"extra", "42"}});
}

UTEST_CASE(parameter_tracker)
{
    std::ostringstream stream_cout, stream_warn, stream_cerr;
    const auto         _ = logger_section_t{stream_cout, stream_warn, stream_cerr};

    auto cmdline      = cmdline_t{"unit testing"};
    auto configurable = configurable_t{};
    UTEST_CHECK_NOTHROW(configurable.register_parameter(parameter_t::make_scalar("fparam", 0.0, LT, 0.5, LT, 1.0)));
    UTEST_CHECK_NOTHROW(configurable.register_parameter(parameter_t::make_integer("iparam", 0, LE, 4, LE, 10)));
    {
        const int   argc    = 1;
        const char* argv[]  = {""};
        const auto  options = cmdline.process(argc, argv);
        auto        tracker = parameter_tracker_t{options};

        UTEST_CHECK_NOTHROW(tracker.setup(configurable));
        UTEST_CHECK_EQUAL(configurable.parameter("iparam").value<int>(), 4);
        UTEST_CHECK_CLOSE(configurable.parameter("fparam").value<double>(), 0.5, 1e-15);
    }
    {
        UTEST_CHECK_EQUAL(stream_cout.str(), "");
        UTEST_CHECK_EQUAL(stream_warn.str(), "");
        UTEST_CHECK_EQUAL(stream_cerr.str(), "");
    }
    {
        const int   argc    = 3;
        const char* argv[]  = {"", "--iparam", "7"};
        const auto  options = cmdline.process(argc, argv);
        auto        tracker = parameter_tracker_t{options};

        UTEST_CHECK_NOTHROW(tracker.setup(configurable));
        UTEST_CHECK_EQUAL(configurable.parameter("iparam").value<int>(), 7);
        UTEST_CHECK_CLOSE(configurable.parameter("fparam").value<double>(), 0.5, 1e-15);
    }
    {
        UTEST_CHECK_EQUAL(stream_cout.str(), "");
        UTEST_CHECK_EQUAL(stream_warn.str(), "");
        UTEST_CHECK_EQUAL(stream_cerr.str(), "");
    }
    {
        const int   argc    = 5;
        const char* argv[]  = {"", "--fparam", "0.42", "--xparam", "42.0"};
        const auto  options = cmdline.process(argc, argv);
        auto        tracker = parameter_tracker_t{options};

        UTEST_CHECK_NOTHROW(tracker.setup(configurable));
        UTEST_CHECK_EQUAL(configurable.parameter("iparam").value<int>(), 7);
        UTEST_CHECK_CLOSE(configurable.parameter("fparam").value<double>(), 0.42, 1e-15);
    }
    {
        UTEST_CHECK_EQUAL(stream_cout.str(), "");
        UTEST_CHECK(ends_with(stream_warn.str(), "parameter \"xparam\" was not used.\n"));
        UTEST_CHECK_EQUAL(stream_cerr.str(), "");
    }
}

UTEST_END_MODULE()
