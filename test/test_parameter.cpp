#include <utest/utest.h>
#include <nano/parameter.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_parameter)

UTEST_CASE(iparam1_LELE)
{
    auto param = sparam1_t{"name", 0, LE, 0, LE, 10};

    UTEST_CHECK_EQUAL(param.get(), 0);

    UTEST_CHECK_NOTHROW(param.set(0));
    UTEST_CHECK_EQUAL(param.get(), 0);

    UTEST_CHECK_NOTHROW(param.set(10));
    UTEST_CHECK_EQUAL(param.get(), 10);

    UTEST_CHECK_NOTHROW(param.set(7));
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(-1), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(11), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(std::numeric_limits<scalar_t>::quiet_NaN()), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get(), 7);

    UTEST_CHECK_THROW(param.set(std::numeric_limits<scalar_t>::infinity()), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get(), 7);
}

UTEST_CASE(iparam2_LELTLE)
{
    auto param = sparam2_t{"name", 0, LE, 1, LT, 2, LE, 10};

    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(1, 1), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(0, 0), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(10, 10), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(-1, 0), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(10, 11), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(3, std::numeric_limits<scalar_t>::quiet_NaN()), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(3, std::numeric_limits<scalar_t>::infinity()), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(std::numeric_limits<scalar_t>::quiet_NaN(), 3), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);

    UTEST_CHECK_THROW(param.set(std::numeric_limits<scalar_t>::infinity(), 3), std::invalid_argument);
    UTEST_CHECK_EQUAL(param.get1(), 1);
    UTEST_CHECK_EQUAL(param.get2(), 2);
}

UTEST_END_MODULE()
