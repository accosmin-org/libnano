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

// TODO: vector(vector) => copy
// TODO: vector = vector => copy
// TODO: vector = pointer => copy

// TODO: pointer(pointer) or pointer(vector) => map
// TODO: pointer = vector => copy (if pointer wise is different)
// TODO: pointer = pointer => copy (if pointer wise is different)

UTEST_BEGIN_MODULE(test_tensor_storage)

UTEST_CASE(vector)
{
    vector_t data0 = vector_t::Constant(7, 0);
    const vector_t data1 = vector_t::Constant(7, 1);
    const vector_t data2 = vector_t::Constant(7, 2);

    {
        const auto vector = vector_storage_t{5};
        UTEST_CHECK_EQUAL(vector.size(), 5);
    }
    {
        const auto vector = vector_storage_t{data1};
        storage_must_match(vector, data1);
    }
    {
        const auto vector1 = vector_storage_t{data1};
        const auto vector2 = vector_storage_t{data1};
        storage_must_match(vector1, data1);
        storage_must_match(vector2, data1);
        storage_must_match(vector1, vector2);
        UTEST_CHECK_NOT_EQUAL(vector1.data(), vector2.data());
    }
    {
        const auto carray = carray_storage_t{data1.data() + 2, 5};
        const auto vector = vector_storage_t{carray};
        storage_must_match(vector, carray);
        storage_must_match(vector, data1.segment(2, 5));
    }
    {
        const auto marray = marray_storage_t{data0.data() + 2, 5};
        const auto vector = vector_storage_t{marray};
        storage_must_match(vector, marray);
        storage_must_match(vector, data0.segment(2, 5));
    }
}

UTEST_END_MODULE()
