#include <nano/tensor/stack.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_tensor_stack)

UTEST_CASE(vector)
{
    const auto stacked_vector =
        stack<int>(9, make_vector<int>(0, 1, 2), tensor_vector_t<int>::Zero(4), tensor_vector_t<int>::Constant(2, -1));

    const auto expected_vector = make_vector<int>(0, 1, 2, 0, 0, 0, 0, -1, -1);

    UTEST_CHECK_EQUAL(stacked_vector, expected_vector);
}

UTEST_CASE(matrix_vertical)
{
    const auto stacked_matrix =
        stack<int>(9, 3, make_vector<int>(0, 1, 2).transpose(), tensor_matrix_t<int>::Identity(3, 3),
                   tensor_matrix_t<int>::Zero(2, 3), make_matrix<int>(3, 9, 8, 7, 6, 5, 4, 3, 2, 1));

    const auto expected_matrix =
        make_matrix<int>(9, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1);

    UTEST_CHECK_EQUAL(stacked_matrix, expected_matrix);
}

UTEST_CASE(matrix_horizontal)
{
    const auto stacked_matrix = stack<int>(5, 3, make_vector<int>(0, 1, 2, 3, 4), tensor_matrix_t<int>::Identity(5, 2));

    const auto expected_matrix = make_matrix<int>(5, 0, 1, 0, 1, 0, 1, 2, 0, 0, 3, 0, 0, 4, 0, 0);

    UTEST_CHECK_EQUAL(stacked_matrix, expected_matrix);
}

UTEST_CASE(matrix_mixed_blocks)
{
    const auto stacked_matrix =
        stack<int>(5, 5, make_matrix<int>(2, 0, 1, 2, 3), make_matrix<int>(2, 9, 8, 7, 6, 5, 4), make_vector<int>(1, 0),
                   make_vector<int>(2, 0), make_vector<int>(3, 0), make_vector<int>(4, 0), make_vector<int>(5, 0),
                   make_vector<int>(5, 6, 7, 8, 9).transpose());

    const auto expected_matrix =
        make_matrix<int>(5, 0, 1, 9, 8, 7, 2, 3, 6, 5, 4, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 5, 6, 7, 8, 9);

    UTEST_CHECK_EQUAL(stacked_matrix, expected_matrix);
}

UTEST_END_MODULE()
