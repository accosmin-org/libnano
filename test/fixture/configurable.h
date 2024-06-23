#include <fstream>
#include <nano/configurable.h>
#include <nano/core/stream.h>
#include <utest/utest.h>

using namespace nano;

template <class tconfigurable>
[[maybe_unused]] static auto check_stream(const tconfigurable& configurable)
{
    // fails to serialize to closed files
    {
        std::ofstream stream;
        UTEST_CHECK_THROW(configurable.write(stream), std::runtime_error);
    }
    {
        std::ifstream stream;
        tconfigurable xconfigurable;
        UTEST_CHECK_THROW(xconfigurable.read(stream), std::runtime_error);
    }

    // serialization to and from in-memory blobs should work using class API
    string_t blob;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(configurable.write(stream));
        blob = stream.str();
    }
    {
        tconfigurable      xconfigurable;
        std::istringstream stream(blob);
        UTEST_CHECK_NOTHROW(xconfigurable.read(stream));
        UTEST_CHECK_EQUAL(xconfigurable.parameters(), configurable.parameters());
    }

    // serialization to and from in-memory blobs should work using functional API
    {
        std::ostringstream ostream;
        UTEST_CHECK_NOTHROW(::nano::write(ostream, configurable));

        tconfigurable      xconfigurable;
        std::istringstream istream(ostream.str());
        UTEST_CHECK_NOTHROW(::nano::read(istream, xconfigurable));
        UTEST_CHECK_EQUAL(xconfigurable.parameters(), configurable.parameters());
        return xconfigurable;
    }
}

[[maybe_unused]] static void config(configurable_t&)
{
}

template <class targ, class... targs>
[[maybe_unused]] static void config(configurable_t& configurable, const char* const param_name, const targ value,
                                    const targs... args)
{
    configurable.parameter(param_name) = value;
    config(configurable, args...);
}
