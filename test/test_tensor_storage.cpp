#include <nano/tensor/storage.h>
#include <utest/utest.h>

using namespace nano;

using vector_t         = tensor_vector_t<double>;
using vector_storage_t = tensor_vector_storage_t<double, 1U>;
using carray_storage_t = tensor_carray_storage_t<double, 1U>;
using marray_storage_t = tensor_marray_storage_t<double, 1U>;

namespace
{
template <typename tlhs, typename trhs>
void storage_must_match(const tlhs& lhs, const trhs& rhs)
{
    UTEST_CHECK_EQUAL(lhs.size(), rhs.size());

    const auto map_lhs = map_vector(lhs.data(), lhs.size());
    const auto map_rhs = map_vector(rhs.data(), rhs.size());
    UTEST_CHECK_CLOSE(map_lhs, map_rhs, 1e-12);
}

auto make_vector_storage(const vector_t& data)
{
    vector_storage_t vector(data.size());
    UTEST_CHECK_EQUAL(vector.size(), data.size());
    map_vector(vector.data(), vector.size()) = data;
    storage_must_match(vector, data);
    return vector;
}
} // namespace

UTEST_BEGIN_MODULE(test_tensor_storage)

UTEST_CASE(vector_init)
{
    vector_t       data0 = vector_t::Constant(7, 0);
    const vector_t data1 = vector_t::Constant(7, 1);

    // vector(dims)
    {
        const auto vector = vector_storage_t{5};
        UTEST_CHECK_EQUAL(vector.size(), 5);
    }
    // vector(dims)
    {
        const auto vector = vector_storage_t{make_dims(5)};
        UTEST_CHECK_EQUAL(vector.size(), 5);
    }
    // vector(vector)
    {
        const auto vector1 = vector_storage_t{5};
        const auto vector2 = vector_storage_t{5};
        UTEST_CHECK_EQUAL(vector1.size(), 5);
        UTEST_CHECK_EQUAL(vector2.size(), 5);
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
    vector_t       data0 = vector_t::Constant(7, 0);
    const vector_t data1 = vector_t::Constant(7, 1);

    // carray(carray)
    {
        const auto carray1 = carray_storage_t{data1.data(), data1.size()};
        const auto carray2 = carray_storage_t{carray1};
        storage_must_match(carray1, data1);
        storage_must_match(carray2, carray2);
        UTEST_CHECK_EQUAL(carray1.data(), data1.data());
        UTEST_CHECK_EQUAL(carray2.data(), data1.data());
    }
    // carray(vector)
    {
        auto vector                              = vector_storage_t{data1.size()};
        map_vector(vector.data(), vector.size()) = data1;
        const auto carray                        = carray_storage_t{vector};
        storage_must_match(carray, vector);
        storage_must_match(carray, data1);
        UTEST_CHECK_NOT_EQUAL(vector.data(), data1.data());
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
    vector_t data0 = vector_t::Constant(7, 0);

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
        auto vector                              = vector_storage_t{data0.size()};
        map_vector(vector.data(), vector.size()) = data0;
        const auto marray                        = marray_storage_t{vector};
        storage_must_match(marray, vector);
        storage_must_match(marray, data0);
        UTEST_CHECK_NOT_EQUAL(vector.data(), data0.data());
        UTEST_CHECK_EQUAL(marray.data(), vector.data());
    }
}

UTEST_CASE(vector_copy)
{
    vector_t       data0 = vector_t::Constant(7, 0);
    const vector_t data1 = vector_t::Constant(5, 1);
    const vector_t data2 = vector_t::Constant(6, 2);

    auto vector = make_vector_storage(data2);
    storage_must_match(vector, data2);
    UTEST_CHECK_NOT_EQUAL(vector.data(), data2.data());

    const auto vother = make_vector_storage(data1);
    vector            = vother;
    storage_must_match(vector, data1);
    UTEST_CHECK_NOT_EQUAL(vector.data(), data1.data());
    UTEST_CHECK_NOT_EQUAL(vector.data(), vother.data());

    const auto carray = carray_storage_t{data2.data(), data2.size()};
    vector            = carray;
    storage_must_match(vector, data2);
    UTEST_CHECK_NOT_EQUAL(vector.data(), data2.data());
    UTEST_CHECK_NOT_EQUAL(vector.data(), carray.data());

    const auto marray = marray_storage_t{data0.data(), data0.size()};
    vector            = marray;
    storage_must_match(vector, data0);
    UTEST_CHECK_NOT_EQUAL(vector.data(), data0.data());
    UTEST_CHECK_NOT_EQUAL(vector.data(), marray.data());
}

UTEST_CASE(marray_copy)
{
    vector_t       data0 = vector_t::Constant(7, 0);
    const vector_t data1 = vector_t::Constant(7, 1);
    const vector_t data2 = vector_t::Constant(7, 2);

    const auto vector = make_vector_storage(data2);
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

UTEST_CASE(resize)
{
    vector_t       data0 = vector_t::Constant(7, 0);
    const vector_t data1 = vector_t::Constant(7, 1);
    const vector_t data2 = vector_t::Constant(7, 2);

    auto vector = make_vector_storage(data2);
    auto marray = marray_storage_t{data0.data(), data0.size()};
    auto carray = carray_storage_t{data1.data(), data1.size()};

    UTEST_CHECK_EQUAL(vector.size(), data2.size());
    UTEST_CHECK_EQUAL(marray.size(), data0.size());
    UTEST_CHECK_EQUAL(carray.size(), data1.size());

    vector.resize(31);
    UTEST_CHECK_EQUAL(vector.size(), 31)

    // marray.resize(31);
    // carray.resize(31);
    //  TODO: to check that this doesn't compile within the same process?! (std::is_invocable doesn't work for =delete)!
}

UTEST_END_MODULE()
