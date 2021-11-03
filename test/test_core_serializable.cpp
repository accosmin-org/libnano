#include <fstream>
#include <utest/utest.h>
#include "fixture/enum.h"
#include <nano/core/stream.h>
#include <nano/core/serializable.h>

using namespace nano;

static auto to_string(const serializable_t& object)
{
    std::ostringstream stream;
    UTEST_REQUIRE_NOTHROW(object.write(stream));
    UTEST_REQUIRE(stream);
    return stream.str();
}

static auto check_stream(const serializable_t& object)
{
    {
        std::ofstream stream;
        UTEST_CHECK_THROW(object.write(stream), std::runtime_error);
    }
    string_t str;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(object.write(stream));
        str = stream.str();
    }
    {
        serializable_t xobject;
        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(xobject.read(stream));
    }
    {
        serializable_t xobject;
        std::ifstream stream;
        UTEST_CHECK_THROW(xobject.read(stream), std::runtime_error);
    }
    {
        std::ostringstream ostream;
        UTEST_CHECK_NOTHROW(::nano::write(ostream, object));

        serializable_t xobject;
        std::istringstream istream(ostream.str());
        UTEST_CHECK_NOTHROW(::nano::read(istream, xobject));
        return xobject;
    }
}

UTEST_BEGIN_MODULE(test_core_serializable)

UTEST_CASE(string)
{
    for (const auto& string : {std::string{}, std::string("stream strings")})
    {
        std::ostringstream ostream;
        UTEST_REQUIRE_NOTHROW(::nano::write(ostream, string));
        UTEST_REQUIRE(ostream);

        const auto ostring = ostream.str();
        UTEST_CHECK_EQUAL(ostring.size(), string.size() + 4);

        std::string istring;
        std::istringstream istream(ostring);
        UTEST_REQUIRE(istream);
        UTEST_REQUIRE_NOTHROW(::nano::read(istream, istring));
        UTEST_REQUIRE(istream);

        UTEST_CHECK_EQUAL(string, istring);

        std::string ifstring;
        std::ifstream ifstream;
        UTEST_REQUIRE(ifstream);
        UTEST_REQUIRE_NOTHROW(::nano::read(ifstream, ifstring));
        UTEST_REQUIRE(!ifstream);
    }
}

UTEST_CASE(serializable_default)
{
    const auto object = serializable_t{};
    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version);
}

UTEST_CASE(serializable_read_const)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4 + 8));

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(object.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version);
}

UTEST_CASE(serializable_read_major)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4 + 8));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[0] = ::nano::major_version - 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(object.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version - 1);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version - 0);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version - 0);
}

UTEST_CASE(serializable_read_minor)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4 + 8));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[1] = ::nano::minor_version - 2; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(object.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version - 0);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version - 2);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version - 0);
}

UTEST_CASE(serializable_read_patch)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4 + 8));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = ::nano::patch_version - 3; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(object.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(object.major_version(), ::nano::major_version - 0);
    UTEST_CHECK_EQUAL(object.minor_version(), ::nano::minor_version - 0);
    UTEST_CHECK_EQUAL(object.patch_version(), ::nano::patch_version - 3);
}

UTEST_CASE(serializable_write_fail)
{
    const auto object = serializable_t{};

    std::ofstream stream;
    UTEST_CHECK_THROW(object.write(stream), std::runtime_error);
}

UTEST_CASE(serializable_read_fail_major)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[0] = ::nano::major_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(object.read(stream), std::runtime_error);
}

UTEST_CASE(serializable_read_fail_minor)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[1] = ::nano::minor_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(object.read(stream), std::runtime_error);
}

UTEST_CASE(serializable_read_fail_patch)
{
    auto object = serializable_t{};

    auto str = to_string(object);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = ::nano::patch_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(object.read(stream), std::runtime_error);
}

UTEST_CASE(no_parameters)
{
    const auto check_params = [] (const serializable_t& object)
    {
        UTEST_CHECK(object.params().empty());
    };

    auto object = serializable_t{};
    check_params(object);

    const auto* const pname = "nonexistent_param_name";
    const auto sname = string_t{"unknown_param_name"};

    UTEST_CHECK_THROW(object.set(pname, int32_t{10}), std::runtime_error);
    UTEST_CHECK_THROW(object.set(pname, int64_t{10}), std::runtime_error);
    UTEST_CHECK_THROW(object.set(pname, 4.2), std::runtime_error);
    UTEST_CHECK_THROW(object.set(pname, enum_type::type1), std::runtime_error);

    UTEST_CHECK_THROW(object.set(sname, int32_t{10}), std::runtime_error);
    UTEST_CHECK_THROW(object.set(sname, int64_t{10}), std::runtime_error);
    UTEST_CHECK_THROW(object.set(sname, 4.2), std::runtime_error);
    UTEST_CHECK_THROW(object.set(sname, enum_type::type1), std::runtime_error);

    UTEST_CHECK_THROW(object.ivalue(pname), std::runtime_error);
    UTEST_CHECK_THROW(object.svalue(pname), std::runtime_error);
    UTEST_CHECK_THROW(object.evalue<enum_type>(pname), std::runtime_error);

    UTEST_CHECK_THROW(object.ivalue(sname), std::runtime_error);
    UTEST_CHECK_THROW(object.svalue(sname), std::runtime_error);
    UTEST_CHECK_THROW(object.evalue<enum_type>(sname), std::runtime_error);

    check_params(check_stream(object));
}

UTEST_CASE(parameters)
{
    const auto check_params = [] (const serializable_t& object)
    {
        UTEST_CHECK_EQUAL(object.params().size(), 6U);

        UTEST_CHECK_EQUAL(object.ivalue("iparam1"), 1);
        UTEST_CHECK_EQUAL(object.ivalue("iparam2"), 2);
        UTEST_CHECK_CLOSE(object.svalue("sparam1"), 1.5, 1e-12);
        UTEST_CHECK_CLOSE(object.svalue("sparam2"), 2.5, 1e-12);
        UTEST_CHECK_CLOSE(object.svalue("sparam3"), 3.5, 1e-12);
        UTEST_CHECK_EQUAL(object.evalue<enum_type>("eparam1"), enum_type::type3);
    };

    auto object = serializable_t{};
    object.register_param(eparam1_t{"eparam1", enum_type::type3});
    object.register_param(iparam1_t{"iparam1", 0, LE, 1, LE, 10});
    object.register_param(iparam1_t{"iparam2", 1, LE, 2, LE, 10});
    object.register_param(sparam1_t{"sparam1", 1.0, LT, 1.5, LT, 2.0});
    object.register_param(sparam1_t{"sparam2", 2.0, LT, 2.5, LT, 3.0});
    object.register_param(sparam1_t{"sparam3", 3.0, LT, 3.5, LT, 4.0});

    check_params(object);
    check_params(check_stream(object));

    UTEST_CHECK_THROW(object.set("eparam1", static_cast<enum_type>(-1)), std::invalid_argument);
    UTEST_CHECK_EQUAL(object.evalue<enum_type>("eparam1"), enum_type::type3);

    UTEST_CHECK_NOTHROW(object.set("eparam1", enum_type::type2));
    UTEST_CHECK_EQUAL(object.evalue<enum_type>("eparam1"), enum_type::type2);

    UTEST_CHECK_NOTHROW(object.set(string_t{"eparam1"}, enum_type::type1));
    UTEST_CHECK_EQUAL(object.evalue<enum_type>(string_t{"eparam1"}), enum_type::type1);

    UTEST_CHECK_THROW(object.set("iparam2", 100), std::runtime_error);
    UTEST_CHECK_EQUAL(object.ivalue("iparam2"), 2);

    UTEST_CHECK_NOTHROW(object.set("iparam2", 3));
    UTEST_CHECK_EQUAL(object.ivalue("iparam2"), 3);

    UTEST_CHECK_NOTHROW(object.set(string_t{"iparam2"}, 7));
    UTEST_CHECK_EQUAL(object.ivalue(string_t{"iparam2"}), 7);

    UTEST_CHECK_THROW(object.set("sparam3", 4.1), std::runtime_error);
    UTEST_CHECK_CLOSE(object.svalue("sparam3"), 3.5, 1e-12);

    UTEST_CHECK_NOTHROW(object.set("sparam3", 3.9));
    UTEST_CHECK_CLOSE(object.svalue("sparam3"), 3.9, 1e-12);

    UTEST_CHECK_NOTHROW(object.set(string_t{"sparam3"}, 3.7));
    UTEST_CHECK_CLOSE(object.svalue(string_t{"sparam3"}), 3.7, 1e-12);
}

UTEST_END_MODULE()
