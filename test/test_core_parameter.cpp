#include <fstream>
#include <utest/utest.h>
#include "fixture/enum.h"
#include <nano/core/parameter.h>

using namespace nano;

static auto to_string(const parameter_t& object)
{
    std::ostringstream stream;
    UTEST_REQUIRE_NOTHROW(stream << object);
    UTEST_REQUIRE(stream);
    return stream.str();
}

static void check_equal(const parameter_t& param, const parameter_t& xparam)
{
    UTEST_CHECK_EQUAL(xparam.name(), param.name());
    UTEST_CHECK_EQUAL(xparam.is_evalue(), param.is_evalue());
    UTEST_CHECK_EQUAL(xparam.is_ivalue(), param.is_ivalue());
    UTEST_CHECK_EQUAL(xparam.is_svalue(), param.is_svalue());
    if (xparam.is_svalue())
    {
        UTEST_CHECK_CLOSE(xparam.svalue(), param.svalue(), 1e-16);
        UTEST_CHECK_CLOSE(xparam.sparam().min(), param.sparam().min(), 1e-16);
        UTEST_CHECK_CLOSE(xparam.sparam().max(), param.sparam().max(), 1e-16);
        UTEST_CHECK_EQUAL(xparam.sparam().minLE(), param.sparam().minLE());
        UTEST_CHECK_EQUAL(xparam.sparam().maxLE(), param.sparam().maxLE());
    }
    else if (xparam.is_ivalue())
    {
        UTEST_CHECK_EQUAL(xparam.ivalue(), param.ivalue());
        UTEST_CHECK_EQUAL(xparam.iparam().min(), param.iparam().min());
        UTEST_CHECK_EQUAL(xparam.iparam().max(), param.iparam().max());
        UTEST_CHECK_EQUAL(xparam.iparam().minLE(), param.iparam().minLE());
        UTEST_CHECK_EQUAL(xparam.iparam().maxLE(), param.iparam().maxLE());
    }
    else
    {
        UTEST_CHECK_EQUAL(xparam.evalue<enum_type>(), param.evalue<enum_type>());
    }
}

static void check_stream(const parameter_t& param)
{
    {
        std::ofstream stream;
        UTEST_CHECK_THROW(param.write(stream), std::runtime_error);
    }
    string_t str;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(param.write(stream));
        str = stream.str();
    }
    {
        parameter_t xparam;
        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(xparam.read(stream));
        check_equal(param, xparam);
    }
    {
        parameter_t xparam;
        reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[0] = 42; // NOLINT
        std::istringstream stream(str);
        UTEST_CHECK_THROW(xparam.read(stream), std::runtime_error);
    }
    {
        parameter_t xparam;
        std::ifstream stream;
        UTEST_CHECK_THROW(xparam.read(stream), std::runtime_error);
    }
    {
        std::ostringstream ostream;
        UTEST_CHECK_NOTHROW(::nano::write(ostream, param));

        parameter_t xparam;
        std::istringstream istream(ostream.str());
        UTEST_CHECK_NOTHROW(::nano::read(istream, xparam));
        check_equal(param, xparam);
    }
}

UTEST_BEGIN_MODULE(test_core_parameter)

UTEST_CASE(eparam1)
{
    auto param = eparam1_t{"name", enum_type::type1};

    UTEST_CHECK_EQUAL(param.name(), "name");
    UTEST_CHECK_EQUAL(param.as<enum_type>(), enum_type::type1);
    UTEST_CHECK_EQUAL(param.get(), scat(enum_type::type1));

    UTEST_CHECK_NOTHROW(param.set(enum_type::type2));
    UTEST_CHECK_EQUAL(param.as<enum_type>(), enum_type::type2);
    UTEST_CHECK_EQUAL(param.get(), scat(enum_type::type2));

    UTEST_CHECK_THROW(param.set(static_cast<enum_type>(-1)), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.as<enum_type>(), enum_type::type2);
    UTEST_CHECK_EQUAL(param.get(), scat(enum_type::type2));

    UTEST_CHECK_NOTHROW(param.set(scat(enum_type::type1)));
    UTEST_CHECK_EQUAL(param.as<enum_type>(), enum_type::type1);
    UTEST_CHECK_EQUAL(param.get(), scat(enum_type::type1));

    UTEST_CHECK_NOTHROW(param = enum_type::type3);
    UTEST_CHECK_EQUAL(param.as<enum_type>(), enum_type::type3);
    UTEST_CHECK_EQUAL(param.get(), scat(enum_type::type3));
}

UTEST_CASE(iparam1_LELE)
{
    auto param = iparam1_t{"name", 0, LE, 0, LE, 10};

    UTEST_CHECK_EQUAL(param.name(), "name");
    UTEST_CHECK_EQUAL(param.get(), 0);
    UTEST_CHECK_EQUAL(param.min(), 0);
    UTEST_CHECK_EQUAL(param.max(), 10);

    UTEST_CHECK_NOTHROW(param.set(0));
    UTEST_CHECK_EQUAL(param.get(), 0);

    UTEST_CHECK_NOTHROW(param.set(10));
    UTEST_CHECK_EQUAL(param.get(), 10);

    UTEST_CHECK_NOTHROW(param.set(7));
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(-1), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(11), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get(), 7);
}

UTEST_CASE(sparam1_LELE)
{
    auto param = sparam1_t{"name", 0, LE, 0, LE, 10};

    UTEST_CHECK_EQUAL(param.name(), "name");
    UTEST_CHECK_EQUAL(param.get(), 0);
    UTEST_CHECK_EQUAL(param.min(), 0);
    UTEST_CHECK_EQUAL(param.max(), 10);

    UTEST_CHECK_NOTHROW(param.set(0));
    UTEST_CHECK_EQUAL(param.get(), 0);

    UTEST_CHECK_NOTHROW(param.set(10));
    UTEST_CHECK_EQUAL(param.get(), 10);

    UTEST_CHECK_NOTHROW(param.set(7));
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(-1), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(11), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(std::numeric_limits<scalar_t>::quiet_NaN()), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(std::numeric_limits<scalar_t>::infinity()), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get(), 7);
}

UTEST_CASE(sparam2_LELTLE)
{
    auto param = sparam2_t{"name", 0, LE, 1, LT, 2, LE, 10};

    UTEST_CHECK_EQUAL(param.name(), "name");
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);
    UTEST_CHECK_EQUAL(param.min(), 0);
    UTEST_CHECK_EQUAL(param.max(), 10);

    UTEST_CHECK_THROW(param.set(1, 1), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(0, 0), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(10, 10), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(-1, 0), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(10, 11), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(3, std::numeric_limits<scalar_t>::quiet_NaN()), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(3, std::numeric_limits<scalar_t>::infinity()), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(std::numeric_limits<scalar_t>::quiet_NaN(), 3), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(std::numeric_limits<scalar_t>::infinity(), 3), std::runtime_error);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);
}

UTEST_CASE(invalid_at_construction)
{
    const auto make1 = [] { return eparam1_t{"name", "type1", strings_t{"typeA", "typeB"}}; };
    const auto make2 = [] { return iparam1_t{"name", 0, LE, -1, LE, 10}; };
    const auto make3 = [] { return iparam1_t{"name", 0, LE, 11, LE, 10}; };
    const auto make4 = [] { return iparam2_t{"name", 0, LE, 3, LT, 1, LE, 10}; };
    const auto make5 = [] { return iparam2_t{"name", 1, LE, 0, LT, 3, LE, 10}; };
    const auto make6 = [] { return iparam2_t{"name", 1, LE, 11, LT, 3, LE, 10}; };
    const auto make7 = [] { return iparam2_t{"name", 1, LE, 11, LT, 12, LE, 10}; };
    const auto make8 = [] { return iparam2_t{"name", 7, LE, 8, LT, 9, LE, 6}; };

    UTEST_CHECK_THROW(make1(), std::runtime_error);
    UTEST_CHECK_THROW(make2(), std::runtime_error);
    UTEST_CHECK_THROW(make3(), std::runtime_error);
    UTEST_CHECK_THROW(make4(), std::runtime_error);
    UTEST_CHECK_THROW(make5(), std::runtime_error);
    UTEST_CHECK_THROW(make6(), std::runtime_error);
    UTEST_CHECK_THROW(make7(), std::runtime_error);
    UTEST_CHECK_THROW(make8(), std::runtime_error);
}

UTEST_CASE(parameter_empty)
{
    auto param = parameter_t{};

    UTEST_CHECK_EQUAL(param.name(), "");
    UTEST_CHECK_EQUAL(param.is_evalue(), true);
    UTEST_CHECK_EQUAL(param.is_ivalue(), false);
    UTEST_CHECK_EQUAL(param.is_svalue(), false);
}

UTEST_CASE(parameter_eparam)
{
    auto param = parameter_t{eparam1_t{"eparam", enum_type::type1}};

    UTEST_CHECK_EQUAL(param.name(), "eparam");
    UTEST_CHECK_EQUAL(param.is_evalue(), true);
    UTEST_CHECK_EQUAL(param.is_ivalue(), false);
    UTEST_CHECK_EQUAL(param.is_svalue(), false);

    UTEST_CHECK_THROW(param.svalue(), std::runtime_error);
    UTEST_CHECK_THROW(param.ivalue(), std::runtime_error);
    UTEST_CHECK_EQUAL(param.evalue<enum_type>(), enum_type::type1);

    UTEST_CHECK_THROW(param.set(int32_t{1}), std::runtime_error);
    UTEST_CHECK_THROW(param.set(int64_t{1}), std::runtime_error);

    UTEST_CHECK_NOTHROW(param.set(enum_type::type2));
    UTEST_CHECK_EQUAL(param.evalue<enum_type>(), enum_type::type2);
    UTEST_CHECK_THROW(param.set(static_cast<enum_type>(-1)), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.evalue<enum_type>(), enum_type::type2);

    check_stream(param);

    UTEST_CHECK_EQUAL(to_string(param), "eparam=type2");
}

UTEST_CASE(parameter_iparam)
{
    auto param = parameter_t{iparam1_t{"iparam", 0, LE, 1, LE, 5}};

    UTEST_CHECK_EQUAL(param.name(), "iparam");
    UTEST_CHECK_EQUAL(param.is_evalue(), false);
    UTEST_CHECK_EQUAL(param.is_ivalue(), true);
    UTEST_CHECK_EQUAL(param.is_svalue(), false);

    UTEST_CHECK_THROW(param.svalue(), std::runtime_error);
    UTEST_CHECK_THROW(param.evalue<enum_type>(), std::runtime_error);
    UTEST_CHECK_EQUAL(param.ivalue(), 1);

    UTEST_CHECK_NOTHROW(param.set(int32_t{0}));
    UTEST_CHECK_EQUAL(param.ivalue(), 0);

    UTEST_CHECK_NOTHROW(param.set(int64_t{5}));
    UTEST_CHECK_EQUAL(param.ivalue(), 5);

    UTEST_CHECK_THROW(param.set(int64_t{7}), std::runtime_error);
    UTEST_CHECK_EQUAL(param.ivalue(), 5);

    UTEST_CHECK_THROW(param.set(int32_t{-1}), std::runtime_error);
    UTEST_CHECK_EQUAL(param.ivalue(), 5);

    UTEST_CHECK_THROW(param.set(scalar_t{0}), std::runtime_error);
    UTEST_CHECK_THROW(param.set(enum_type::type1), std::runtime_error);

    check_stream(param);
    check_stream(parameter_t{iparam1_t{"iparam", 0, LE, 1, LT, 5}});
    check_stream(parameter_t{iparam1_t{"iparam", 0, LT, 1, LE, 5}});
    check_stream(parameter_t{iparam1_t{"iparam", 0, LT, 1, LT, 5}});

    UTEST_CHECK_EQUAL(to_string(param), "iparam=5");
}

UTEST_CASE(parameter_sparam)
{
    auto param = parameter_t{sparam1_t{"sparam", 0, LE, 1, LE, 5}};

    UTEST_CHECK_EQUAL(param.name(), "sparam");
    UTEST_CHECK_EQUAL(param.is_evalue(), false);
    UTEST_CHECK_EQUAL(param.is_ivalue(), false);
    UTEST_CHECK_EQUAL(param.is_svalue(), true);

    UTEST_CHECK_CLOSE(param.svalue(), 1.0, 1e-12);
    UTEST_CHECK_THROW(param.evalue<enum_type>(), std::runtime_error);
    UTEST_CHECK_THROW(param.ivalue(), std::runtime_error);

    UTEST_CHECK_NOTHROW(param.set(0.1));
    UTEST_CHECK_CLOSE(param.svalue(), 0.1, 1e-12);

    UTEST_CHECK_THROW(param.set(-1.1), std::runtime_error);
    UTEST_CHECK_CLOSE(param.svalue(), 0.1, 1e-12);

    UTEST_CHECK_THROW(param.set(5.1), std::runtime_error);
    UTEST_CHECK_CLOSE(param.svalue(), 0.1, 1e-12);

    UTEST_CHECK_NOTHROW(param.set(int32_t{0}));
    UTEST_CHECK_CLOSE(param.svalue(), 0.0, 1e-12);

    UTEST_CHECK_NOTHROW(param.set(int64_t{1}));
    UTEST_CHECK_CLOSE(param.svalue(), 1.0, 1e-12);

    UTEST_CHECK_THROW(param.set(enum_type::type1), std::runtime_error);

    check_stream(param);
    check_stream(parameter_t{sparam1_t{"sparam", 0, LE, 1, LT, 5}});
    check_stream(parameter_t{sparam1_t{"sparam", 0, LT, 1, LE, 5}});
    check_stream(parameter_t{sparam1_t{"sparam", 0, LT, 1, LT, 5}});

    UTEST_CHECK_EQUAL(to_string(param), "sparam=1");
}

UTEST_END_MODULE()
