#include <nano/tensor/algorithm.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
template <class toperator, class... ttensors>
auto copy_remove_if(const toperator& op, ttensors... tensors)
{
    const auto count = nano::remove_if(op, tensors...);
    return std::make_tuple(count, tensors...);
}
} // namespace

UTEST_BEGIN_MODULE()

UTEST_CASE(vector)
{
    const auto xdata = make_vector<int>(0, 1, 2, 3, 4, 5, 6, 7);
    {
        const auto op = [&](const tensor_size_t) { return true; };

        const auto [size, data] = copy_remove_if(op, xdata);
        UTEST_CHECK_EQUAL(size, 0);
        UTEST_CHECK_EQUAL(data, make_vector<int>(0, 1, 2, 3, 4, 5, 6, 7));
    }
    {
        const auto op = [&](const tensor_size_t i) { return xdata(i) % 3 == 2; };

        const auto [size, data] = copy_remove_if(op, xdata);
        UTEST_CHECK_EQUAL(size, 6);
        UTEST_CHECK_EQUAL(data, make_vector<int>(0, 1, 3, 4, 6, 7, 6, 7));
    }
    {
        const auto op = [&](const tensor_size_t i) { return xdata(i) > 10; };

        const auto [size, data] = copy_remove_if(op, xdata);
        UTEST_CHECK_EQUAL(size, 8);
        UTEST_CHECK_EQUAL(data, make_vector<int>(0, 1, 2, 3, 4, 5, 6, 7));
    }
}

UTEST_CASE(mixed)
{
    const auto xvec = make_vector<int>(0, 1, 2, 3, 4, 5, 6, 7);
    const auto xmat = make_matrix<int>(8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    {
        const auto op = [&](const tensor_size_t) { return true; };

        const auto [size, vec, mat] = copy_remove_if(op, xvec, xmat);
        UTEST_CHECK_EQUAL(size, 0);
        UTEST_CHECK_EQUAL(vec, make_vector<int>(0, 1, 2, 3, 4, 5, 6, 7));
        UTEST_CHECK_EQUAL(mat, make_matrix<int>(8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    }
    {
        const auto op = [&](const tensor_size_t i) { return i == 0; };

        const auto [size, vec, mat] = copy_remove_if(op, xvec, xmat);
        UTEST_CHECK_EQUAL(size, 7);
        UTEST_CHECK_EQUAL(vec, make_vector<int>(1, 2, 3, 4, 5, 6, 7, 7));
        UTEST_CHECK_EQUAL(mat, make_matrix<int>(8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 15));
    }
    {
        const auto op = [&](const tensor_size_t i) { return i < 3 || i > 5; };

        const auto [size, vec, mat] = copy_remove_if(op, xvec, xmat);
        UTEST_CHECK_EQUAL(size, 3);
        UTEST_CHECK_EQUAL(vec, make_vector<int>(3, 4, 5, 3, 4, 5, 6, 7));
        UTEST_CHECK_EQUAL(mat, make_matrix<int>(8, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    }
    {
        const auto op = [&](const tensor_size_t i) { return i > 5; };

        const auto [size, vec, mat] = copy_remove_if(op, xvec, xmat);
        UTEST_CHECK_EQUAL(size, 6);
        UTEST_CHECK_EQUAL(vec, make_vector<int>(0, 1, 2, 3, 4, 5, 6, 7));
        UTEST_CHECK_EQUAL(mat, make_matrix<int>(8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    }
    {
        const auto op = [&](const tensor_size_t i) { return xmat(i, 0) + xmat(i, 1) == 9; };

        const auto [size, vec, mat] = copy_remove_if(op, xvec, xmat);
        UTEST_CHECK_EQUAL(size, 7);
        UTEST_CHECK_EQUAL(vec, make_vector<int>(0, 1, 3, 4, 5, 6, 7, 7));
        UTEST_CHECK_EQUAL(mat, make_matrix<int>(8, 0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 15));
    }
    {
        const auto op = [&](const tensor_size_t i) { return (xmat(i, 0) + xmat(i, 1) == 9) || (xvec(i) > 4); };

        const auto [size, vec, mat] = copy_remove_if(op, xvec, xmat);
        UTEST_CHECK_EQUAL(size, 4);
        UTEST_CHECK_EQUAL(vec, make_vector<int>(0, 1, 3, 4, 4, 5, 6, 7));
        UTEST_CHECK_EQUAL(mat, make_matrix<int>(8, 0, 1, 2, 3, 6, 7, 8, 9, 8, 9, 10, 11, 12, 13, 14, 15));
    }
}

UTEST_END_MODULE()
