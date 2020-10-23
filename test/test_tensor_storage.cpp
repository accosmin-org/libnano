#include <utest/utest.h>
#include <nano/tensor/storage2.h>

using namespace nano;

using vector_t = tensor_vector_t<double>;

using vector_storage_t = tensor_vector_storage_t<double>;
using carray_storage_t = tensor_carray_storage_t<double>;
using marray_storage_t = tensor_marray_storage_t<double>;

template <typename tlhs, typename trhs>
static void storage_must_match(const tlhs& lhs, const trhs& rhs)
{
    UTEST_CHECK_EQUAL(lhs.size(), rhs.size());

    const auto map_lhs = map_vector(lhs.data(), lhs.size());
    const auto map_rhs = map_vector(rhs.data(), rhs.size());
    UTEST_CHECK_EIGEN_CLOSE(map_lhs, map_rhs, 1e-12);
}

static vector_t data0 = vector_t::Constant(7, 0);
static const vector_t data1 = vector_t::Constant(7, 1);
static const vector_t data2 = vector_t::Constant(7, 2);

UTEST_BEGIN_MODULE(test_tensor_storage)

UTEST_CASE(vector_init)
{
    // vector(size)
    {
        const auto vector = vector_storage_t{5};
        UTEST_CHECK_EQUAL(vector.size(), 5);
    }
    // vector(eigen)
    {
        const auto vector1 = vector_storage_t{data1};
        const auto vector2 = vector_storage_t{data1};
        storage_must_match(vector1, data1);
        storage_must_match(vector2, data1);
        storage_must_match(vector1, vector2);
        UTEST_CHECK_NOT_EQUAL(vector1.data(), data1.data());
        UTEST_CHECK_NOT_EQUAL(vector2.data(), data1.data());
        UTEST_CHECK_NOT_EQUAL(vector1.data(), vector2.data());
    }
    // vector(vector)
    {
        const auto vector1 = vector_storage_t{data1};
        const auto vector2 = vector_storage_t{vector1};
        storage_must_match(vector1, data1);
        storage_must_match(vector2, data1);
        storage_must_match(vector1, vector2);
        UTEST_CHECK_NOT_EQUAL(vector1.data(), data1.data());
        UTEST_CHECK_NOT_EQUAL(vector2.data(), data1.data());
        UTEST_CHECK_NOT_EQUAL(vector1.data(), vector2.data());
    }
    // vector(carray)
    {
        const auto carray = carray_storage_t{data1.data() + 2, 5};
        const auto vector = vector_storage_t{carray};
        storage_must_match(vector, carray);
        storage_must_match(vector, data1.segment(2, 5));
        UTEST_CHECK_EQUAL(carray.data(), data1.data() + 2);
        UTEST_CHECK_NOT_EQUAL(vector.data(), carray.data());
    }
    // vector(marray)
    {
        const auto marray = marray_storage_t{data0.data() + 1, 4};
        const auto vector = vector_storage_t{marray};
        storage_must_match(vector, marray);
        storage_must_match(vector, data0.segment(1, 4));
        UTEST_CHECK_EQUAL(marray.data(), data0.data() + 1);
        UTEST_CHECK_NOT_EQUAL(vector.data(), marray.data());
    }
}

UTEST_CASE(carray_init)
{
    // carray(carray)
    {
        const auto carray1 = carray_storage_t{data0.data(), data0.size()};
        const auto carray2 = carray_storage_t{carray1};
        storage_must_match(carray1, data0);
        storage_must_match(carray2, carray2);
        UTEST_CHECK_EQUAL(carray1.data(), data0.data());
        UTEST_CHECK_EQUAL(carray2.data(), data0.data());
    }
    // carray(vector)
    {
        const auto vector = vector_storage_t{data0};
        const auto carray = carray_storage_t{vector};
        storage_must_match(carray, vector);
        storage_must_match(carray, data0);
        UTEST_CHECK_NOT_EQUAL(vector.data(), data0.data());
        UTEST_CHECK_EQUAL(carray.data(), vector.data());
    }
    // carray(marray)
    {
        const auto marray = marray_storage_t{data0.data() + 3, 4};
        const auto carray = carray_storage_t{marray};
        storage_must_match(carray, marray);
        storage_must_match(carray, data0.segment(3, 4));
        UTEST_CHECK_EQUAL(marray.data(), data0.data() + 3);
        UTEST_CHECK_EQUAL(carray.data(), marray.data());
    }
}

UTEST_CASE(marray_init)
{
    // marray(marray)
    {
        const auto marray1 = marray_storage_t{data0.data(), data0.size()};
        const auto marray2 = marray_storage_t{marray1};
        storage_must_match(marray1, data0);
        storage_must_match(marray2, marray2);
        UTEST_CHECK_EQUAL(marray1.data(), data0.data());
        UTEST_CHECK_EQUAL(marray2.data(), data0.data());
    }
    // marray(vector)
    {
        auto vector = vector_storage_t{data0};
        const auto marray = marray_storage_t{vector};
        storage_must_match(marray, vector);
        storage_must_match(marray, data0);
        UTEST_CHECK_NOT_EQUAL(vector.data(), data0.data());
        UTEST_CHECK_EQUAL(marray.data(), vector.data());
    }
}

UTEST_CASE(vector_copy)
{
    auto vector = vector_storage_t{data2};
    storage_must_match(vector, data2);
    UTEST_CHECK_NOT_EQUAL(vector.data(), data2.data());

    const auto vother = vector_storage_t{data1};
    vector = vother;
    storage_must_match(vector, data1);
    UTEST_CHECK_NOT_EQUAL(vector.data(), data1.data());
    UTEST_CHECK_NOT_EQUAL(vector.data(), vother.data());

    const auto carray = carray_storage_t{data2.data(), data2.size()};
    vector = carray;
    storage_must_match(vector, data2);
    UTEST_CHECK_NOT_EQUAL(vector.data(), data2.data());
    UTEST_CHECK_NOT_EQUAL(vector.data(), carray.data());

    const auto marray = marray_storage_t{data0.data(), data0.size()};
    vector = marray;
    storage_must_match(vector, data0);
    UTEST_CHECK_NOT_EQUAL(vector.data(), data0.data());
    UTEST_CHECK_NOT_EQUAL(vector.data(), marray.data());
}

UTEST_CASE(marray_copy)
{
    const auto vector = vector_storage_t{data2};
    const auto marray = marray_storage_t{data0.data(), data0.size()};
    const auto carray = carray_storage_t{data1.data(), data1.size()};

    storage_must_match(marray, data0);
    storage_must_match(carray, data1);
    storage_must_match(vector, data2);

    UTEST_CHECK_EQUAL(marray.data(), data0.data());
    UTEST_CHECK_EQUAL(carray.data(), data1.data());
    UTEST_CHECK_NOT_EQUAL(vector.data(), data2.data());

    vector_t data = vector_t::Constant(7, -1);

    auto array = marray_storage_t{data.data(), data.size()};
    storage_must_match(array, data);
    UTEST_CHECK_EQUAL(array.data(), data.data());

    array = vector;
    storage_must_match(array, data2);
    UTEST_CHECK_EQUAL(array.data(), data.data());

    array = marray;
    storage_must_match(array, data0);
    UTEST_CHECK_EQUAL(array.data(), data.data());

    array = carray;
    storage_must_match(array, data1);
    UTEST_CHECK_EQUAL(array.data(), data.data());
}

UTEST_END_MODULE()
