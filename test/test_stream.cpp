#include <fstream>
#include <sstream>
#include <utest/utest.h>
#include <nano/stream.h>

using namespace nano;

auto to_string(const serializable_t& object)
{
    std::ostringstream stream;
    UTEST_REQUIRE_NOTHROW(object.write(stream));
    UTEST_REQUIRE(stream);
    return stream.str();
}

UTEST_BEGIN_MODULE(test_stream)

UTEST_CASE(string)
{
    for (const auto& string : {std::string{}, std::string("stream strings")})
    {
        std::ostringstream ostream;
        UTEST_REQUIRE_NOTHROW(::nano::detail::write(ostream, string));
        UTEST_REQUIRE(ostream);

        const auto ostring = ostream.str();
        UTEST_CHECK_EQUAL(ostring.size(), string.size() + 4);

        std::string istring;
        std::istringstream istream(ostring);
        UTEST_REQUIRE(istream);
        UTEST_REQUIRE_NOTHROW(::nano::detail::read(istream, istring));
        UTEST_REQUIRE(istream);

        UTEST_CHECK_EQUAL(string, istring);

        std::string ifstring;
        std::ifstream ifstream;
        UTEST_REQUIRE(ifstream);
        UTEST_REQUIRE_NOTHROW(::nano::detail::read(ifstream, ifstring));
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
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4));

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
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4));
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
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4));
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
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4));
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

UTEST_END_MODULE()
