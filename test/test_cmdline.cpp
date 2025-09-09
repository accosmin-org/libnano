#include <filesystem>
#include <fstream>
#include <nano/core/cmdline.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
void check(const cmdvalues_t& values, const cmdvalues_t& expected_values)
{
    UTEST_CHECK_EQUAL(values.size(), expected_values.size());

    for (const auto& [name, expected_value] : expected_values)
    {
        const auto it = values.find(name);
        UTEST_REQUIRE(it != values.end());

        const auto& value = it->second;

        UTEST_REQUIRE_EQUAL(value.has_value(), expected_value.has_value());
        UTEST_CHECK_EQUAL(value.m_index, expected_value.m_index);

        if (value.has_value())
        {
            UTEST_CHECK_EQUAL(value.value(), expected_value.value());
        }
    }
}

void check(const cmdresult_t& result, const cmdvalues_t& expected_values)
{
    check(result.m_values, expected_values);
}
} // namespace

UTEST_BEGIN_MODULE()

UTEST_CASE(handle_help)
{
    const auto cmdline = cmdline_t{"unit testing"};
    for (const auto* const cmdstr : {"-h", "--help"})
    {
        const auto options = cmdline.process(cmdstr);
        check(options, {
                           {    "-h", {{}, 0}},
                           {"--help", {{}, 0}},
        });

        std::stringstream stream;
        UTEST_CHECK(cmdline.handle(options, stream));
        UTEST_CHECK_EQUAL(stream.str(), R"XXX(unit testing
  -h,--help        print usage
  -v,--version     print library's version
  -g,--git-hash    print library's git commit hash
)XXX");
    }
}

UTEST_CASE(handle_version)
{
    const auto cmdline = cmdline_t{"unit testing"};
    for (const auto* const cmdstr : {"-v", "--version"})
    {
        const auto options = cmdline.process(cmdstr);
        check(options, {
                           {       "-v", {{}, 1}},
                           {"--version", {{}, 1}},
        });

        std::stringstream stream;
        UTEST_CHECK(cmdline.handle(options, stream));
        UTEST_CHECK_EQUAL(stream.str(), scat(major_version, ".", minor_version, ".", patch_version, "\n"));
    }
}

UTEST_CASE(handle_githash)
{
    const auto cmdline = cmdline_t{"unit testing"};
    for (const auto* const cmdstr : {"-g", "--git-hash"})
    {
        const auto options = cmdline.process(cmdstr);
        check(options, {
                           {        "-g", {{}, 2}},
                           {"--git-hash", {{}, 2}},
        });

        std::stringstream stream;
        UTEST_CHECK(cmdline.handle(options, stream));
        UTEST_CHECK_EQUAL(stream.str(), scat(git_commit_hash, "\n"));
    }
}

UTEST_CASE(complex_usage)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("-d,--doit", "do something important if set"));
    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "version number", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("--iterations", "number of iterations", 100));

    std::stringstream stream;
    UTEST_CHECK(cmdline.handle(cmdline.process("-h"), stream));
    UTEST_CHECK_EQUAL(stream.str(), R"XXX(unit testing
  -h,--help             print usage
  -v,--version          print library's version
  -g,--git-hash         print library's git commit hash
  -d,--doit             do something important if set
  -x,--xversion(0.3)    version number
  --iterations(100)     number of iterations
)XXX");
}

UTEST_CASE(parse_chars)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("--trials", "number of trials"));
    UTEST_CHECK_NOTHROW(cmdline.add("--iterations", "number of iterations"));

    const int   argc   = 3;
    const char* argv[] = {"", "-x", "0.3.1"};

    check(cmdline.process(argc, argv), {
                                           {        "-x", {"0.3.1", 3}},
                                           {"--xversion", {"0.3.1", 3}},
    });
}

UTEST_CASE(parse_string)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("--doit", "do something important if set"));
    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "version", "0.3"));
    UTEST_CHECK_NOTHROW(cmdline.add("--iterations", "number of iterations", 127));

    check(cmdline.process("--help --iterations 7"), {
                                                        {          "-h",    {{}, 0}},
                                                        {      "--help",    {{}, 0}},
                                                        {          "-x", {"0.3", 4}},
                                                        {  "--xversion", {"0.3", 4}},
                                                        {"--iterations",   {"7", 5}}
    });

    check(cmdline.process("-x 1.0 --extra1 value1 --extra2 value2 -y value3"), {
                                                                                   {          "-x", {"1.0", 4}},
                                                                                   {  "--xversion", {"1.0", 4}},
                                                                                   {"--iterations", {"127", 5}},
                                                                                   {    "--extra1", {"value1"}},
                                                                                   {    "--extra2", {"value2"}},
                                                                                   {          "-y", {"value3"}}
    });
}

UTEST_CASE(invalid_options)
{
    auto cmdline = cmdline_t{"unit testing"};

    UTEST_CHECK_THROW(cmdline.add("", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("x", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-x", ""), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-x,xxx", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-x,--x,-x", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-,--x,-x", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("--,-x", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("---,--x,-x", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("x,xxx", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-x, --xxx", "description"), std::runtime_error);
}

UTEST_CASE(error_duplicate_options)
{
    auto cmdline = cmdline_t{"unit testing"};

    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "description"));
    UTEST_CHECK_THROW(cmdline.add("-x,--xversion", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-x,--xwersion", "description"), std::runtime_error);
    UTEST_CHECK_THROW(cmdline.add("-w,--xversion", "description"), std::runtime_error);
}

UTEST_CASE(invalid_arg_expecting_option_name_with_dash)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("--iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "x", "--xversion", "7"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(invalid_arg_invalid_dash_option_name)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("--iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "--xversion", "7", "-", "--xversion", "13"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(invalid_arg_invalid_double_dash_option_name)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("--iterations", "number of iterations", "127"));

    const int   argc   = 4;
    const char* argv[] = {"", "--xversion", "11", "--"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(invalid_arg_expecting_option_name_with_dash)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("--iterations", "number of iterations", "127"));

    const int   argc   = 5;
    const char* argv[] = {"", "-x", "--extra", "7", "17"};

    UTEST_CHECK_THROW(cmdline.process(argc, argv), std::runtime_error);
}

UTEST_CASE(parse_config_file)
{
    auto cmdline = cmdline_t{"unit testing"};
    UTEST_CHECK_NOTHROW(cmdline.add("-x,--xversion", "version"));
    UTEST_CHECK_NOTHROW(cmdline.add("--iterations", "number of iterations", "127"));

    const auto path = std::filesystem::temp_directory_path() / "libnano.config.tmp";
    {
        std::ofstream out(path);
        out << "-x\n";
        out << "--iterations 102\n";
        out << "--extra str\n";
        out << "-flag\n";
    }

    const auto options = cmdline.process_config_file(path);
    check(options, {
                       {          "-x",    {{}, 3}},
                       {  "--xversion",    {{}, 3}},
                       {"--iterations", {"102", 4}},
                       {     "--extra",    {"str"}},
                       {       "-flag",         {}}
    });

    UTEST_CHECK(options.has("-x"));
    UTEST_CHECK(options.has("--iterations"));
    UTEST_CHECK(!options.has("iterations"));
    UTEST_CHECK(options.has("--extra"));
    UTEST_CHECK(options.has("-flag"));
    UTEST_CHECK(!options.has("extra"));

    UTEST_CHECK(!options.has_value("-x"));
    UTEST_CHECK(options.has_value("--iterations"));
    UTEST_CHECK(!options.has_value("iterations"));
    UTEST_CHECK(options.has_value("--extra"));
    UTEST_CHECK(!options.has_value("-flag"));
    UTEST_CHECK(!options.has_value("extra"));

    UTEST_CHECK_THROW(options.get("x"), std::runtime_error);
    UTEST_CHECK_THROW(options.get("extra"), std::runtime_error);
    UTEST_CHECK_THROW(options.get("iterations"), std::runtime_error);
    UTEST_CHECK_THROW(options.get("-x"), std::runtime_error);
    UTEST_CHECK_THROW(options.get("--xversion"), std::runtime_error);
    UTEST_CHECK_THROW(options.get("-flag"), std::runtime_error);

    UTEST_CHECK_NOTHROW(options.get("--extra"));
    UTEST_CHECK_EQUAL(options.get("--extra"), "str");
    UTEST_CHECK_EQUAL(options.get<string_t>("--extra"), "str");
    UTEST_CHECK_THROW(options.get<int>("--extra"), std::invalid_argument);

    UTEST_CHECK_NOTHROW(options.get("--iterations"));
    UTEST_CHECK_EQUAL(options.get("--iterations"), "102");
    UTEST_CHECK_EQUAL(options.get<string_t>("--iterations"), "102");
    UTEST_CHECK_EQUAL(options.get<int>("--iterations"), 102);
}

UTEST_CASE(cmdconfig)
{
    std::ostringstream stream;

    auto cmdline      = cmdline_t{"unit testing"};
    auto configurable = configurable_t{};
    UTEST_CHECK_NOTHROW(configurable.register_parameter(parameter_t::make_scalar("fparam", 0.0, LT, 0.5, LT, 1.0)));
    UTEST_CHECK_NOTHROW(configurable.register_parameter(parameter_t::make_integer("iparam", 0, LE, 4, LE, 10)));
    {
        const int   argc    = 1;
        const char* argv[]  = {""};
        const auto  options = cmdline.process(argc, argv);
        auto        rconfig = cmdconfig_t{options, make_stream_logger(stream)};

        UTEST_CHECK_NOTHROW(rconfig.setup(configurable));
        UTEST_CHECK_EQUAL(configurable.parameter("iparam").value<int>(), 4);
        UTEST_CHECK_CLOSE(configurable.parameter("fparam").value<double>(), 0.5, 1e-15);
    }
    {
        UTEST_CHECK_EQUAL(stream.str(), "");
    }
    {
        const int   argc    = 3;
        const char* argv[]  = {"", "--iparam", "7"};
        const auto  options = cmdline.process(argc, argv);
        auto        rconfig = cmdconfig_t{options, make_stream_logger(stream)};

        UTEST_CHECK_NOTHROW(rconfig.setup(configurable));
        UTEST_CHECK_EQUAL(configurable.parameter("iparam").value<int>(), 7);
        UTEST_CHECK_CLOSE(configurable.parameter("fparam").value<double>(), 0.5, 1e-15);
    }
    {
        UTEST_CHECK_EQUAL(stream.str(), "");
    }
    {
        const int   argc    = 5;
        const char* argv[]  = {"", "--fparam", "0.42", "--xparam", "42.0"};
        const auto  options = cmdline.process(argc, argv);
        auto        rconfig = cmdconfig_t{options, make_stream_logger(stream)};

        UTEST_CHECK_NOTHROW(rconfig.setup(configurable));
        UTEST_CHECK_EQUAL(configurable.parameter("iparam").value<int>(), 7);
        UTEST_CHECK_CLOSE(configurable.parameter("fparam").value<double>(), 0.42, 1e-15);
    }
    {
        UTEST_CHECK(ends_with(stream.str(), "parameter '--xparam' was not used.\n"));
    }
}

UTEST_END_MODULE()
