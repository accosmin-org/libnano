#include <utest/utest.h>
#include <nano/tensor/integral.h>

using namespace nano;

template <typename tscalar, typename tstorage>
static auto check(const tensor_t<tstorage, 1>& tensor,
    const tensor_size_t index0)
{
    tscalar sum = 0;
    for (tensor_size_t i0 = 0; i0 <= index0; ++ i0)
    {
        sum += tensor(i0);
    }

    return sum;
}

template <typename tscalar, typename tstorage>
static auto check(const tensor_t<tstorage, 2>& tensor,
    const tensor_size_t index0, const tensor_size_t index1)
{
    tscalar sum = 0;
    for (tensor_size_t i0 = 0; i0 <= index0; ++ i0)
    {
        for (tensor_size_t i1 = 0; i1 <= index1; ++ i1)
        {
            sum += tensor(i0, i1);
        }
    }

    return sum;
}

template <typename tscalar, typename tstorage>
static auto check(const tensor_t<tstorage, 3>& tensor,
    const tensor_size_t index0, const tensor_size_t index1, const tensor_size_t index2)
{
    tscalar sum = 0;
    for (tensor_size_t i0 = 0; i0 <= index0; ++ i0)
    {
        for (tensor_size_t i1 = 0; i1 <= index1; ++ i1)
        {
            for (tensor_size_t i2 = 0; i2 <= index2; ++ i2)
            {
                sum += tensor(i0, i1, i2);
            }
        }
    }

    return sum;
}

template <typename tscalar, typename tstorage>
static auto check(const tensor_t<tstorage, 4>& tensor,
    const tensor_size_t index0, const tensor_size_t index1, const tensor_size_t index2, const tensor_size_t index3)
{
    tscalar sum = 0;
    for (tensor_size_t i0 = 0; i0 <= index0; ++ i0)
    {
        for (tensor_size_t i1 = 0; i1 <= index1; ++ i1)
        {
            for (tensor_size_t i2 = 0; i2 <= index2; ++ i2)
            {
                for (tensor_size_t i3 = 0; i3 <= index3; ++ i3)
                {
                    sum += tensor(i0, i1, i2, i3);
                }
            }
        }
    }

    return sum;
}

UTEST_BEGIN_MODULE(test_tensor_integral)

UTEST_CASE(integral1d)
{
    tensor_mem_t<int32_t, 1> xtensor(11);
    tensor_mem_t<int64_t, 1> itensor(11);

    xtensor.random();
    itensor.random();
    nano::integral(xtensor, itensor);

    UTEST_REQUIRE_EQUAL(xtensor.dims(), itensor.dims());
    for (tensor_size_t i0 = 0; i0 < xtensor.size<0>(); ++ i0)
    {
        UTEST_CHECK_EQUAL(itensor(i0), check<int64_t>(xtensor, i0));
    }
}

UTEST_CASE(integral2d)
{
    tensor_mem_t<int32_t, 2> xtensor(9, 11);
    tensor_mem_t<int64_t, 2> itensor(9, 11);

    xtensor.random();
    itensor.random();
    nano::integral(xtensor, itensor);

    UTEST_REQUIRE_EQUAL(xtensor.dims(), itensor.dims());
    for (tensor_size_t i0 = 0; i0 < xtensor.size<0>(); ++ i0)
    {
        for (tensor_size_t i1 = 0; i1 < xtensor.size<1>(); ++ i1)
        {
            UTEST_CHECK_EQUAL(itensor(i0, i1), check<int64_t>(xtensor, i0, i1));
        }
    }
}

UTEST_CASE(integral3d)
{
    tensor_mem_t<int32_t, 3> xtensor(7, 9, 11);
    tensor_mem_t<int64_t, 3> itensor(7, 9, 11);

    xtensor.random();
    itensor.random();
    nano::integral(xtensor, itensor);

    UTEST_REQUIRE_EQUAL(xtensor.dims(), itensor.dims());
    for (tensor_size_t i0 = 0; i0 < xtensor.size<0>(); ++ i0)
    {
        for (tensor_size_t i1 = 0; i1 < xtensor.size<1>(); ++ i1)
        {
            for (tensor_size_t i2 = 0; i2 < xtensor.size<2>(); ++ i2)
            {
                UTEST_CHECK_EQUAL(itensor(i0, i1, i2), check<int64_t>(xtensor, i0, i1, i2));
            }
        }
    }
}

UTEST_CASE(integral4d)
{
    tensor_mem_t<int32_t, 4> xtensor(5, 7, 9, 11);
    tensor_mem_t<int64_t, 4> itensor(5, 7, 9, 11);

    xtensor.random();
    itensor.random();
    nano::integral(xtensor, itensor);

    UTEST_REQUIRE_EQUAL(xtensor.dims(), itensor.dims());
    for (tensor_size_t i0 = 0; i0 < xtensor.size<0>(); ++ i0)
    {
        for (tensor_size_t i1 = 0; i1 < xtensor.size<1>(); ++ i1)
        {
            for (tensor_size_t i2 = 0; i2 < xtensor.size<2>(); ++ i2)
            {
                for (tensor_size_t i3 = 0; i3 < xtensor.size<3>(); ++ i3)
                {
                    UTEST_CHECK_EQUAL(itensor(i0, i1, i2, i3), check<int64_t>(xtensor, i0, i1, i2, i3));
                }
            }
        }
    }
}

UTEST_END_MODULE()
