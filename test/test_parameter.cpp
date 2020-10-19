#include <utest/utest.h>
#include "fixture/enum.h"
#include <nano/parameter.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_parameter)

UTEST_CASE(eparam1)
{
    auto param = eparam1_t{"name", enum_type::type1};

    UTEST_CHECK_EQUAL(param.name(), "name");
    UTEST_CHECK_EQUAL(param.as<enum_type>(), enum_type::type1);
    UTEST_CHECK_EQUAL(param.get(), static_cast<int64_t>(enum_type::type1));

    UTEST_CHECK_NOTHROW(param.set(enum_type::type2));
    UTEST_CHECK_EQUAL(param.as<enum_type>(), enum_type::type2);
    UTEST_CHECK_EQUAL(param.get(), static_cast<int64_t>(enum_type::type2));

    UTEST_CHECK_THROW(param.set(static_cast<enum_type>(-1)), std::runtime_error);
    UTEST_CHECK_EQUAL(param.as<enum_type>(), enum_type::type2);
    UTEST_CHECK_EQUAL(param.get(), static_cast<int64_t>(enum_type::type2));
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
    const auto make1 = [] { return eparam1_t{"name", static_cast<enum_type>(-1)}; };
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

UTEST_END_MODULE()
