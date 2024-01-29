#include <nano/tensor/algorithm.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_tensor_algorithm)

UTEST_CASE(vector)
{
    const auto data = make_vector<int>(0, 1, 2, 3, 4, 5, 6, 7);
    {
        const auto op = [&](const tensor_size_t index) { return data(index) % 3 == 2; };

        const auto expected_size = 6;
        const auto expected_data = make_vector<int>(0, 1, 3, 4, 6, 7, 7, 7);

        auto       copy = data;
        const auto size = remove_if(op, copy);
        UTEST_CHECK_EQUAL(size, expected_size);
        UTEST_CHECK_EQUAL(data, expected_data);
    }
    {
        const auto op = [&](const tensor_size_t index) { return data(index) > 10; };

        const auto expected_size = 8;
        const auto expected_data = make_vector<int>(0, 1, 2, 3, 4, 5, 6, 7);

        auto       copy = data;
        const auto size = remove_if(op, copy);
        UTEST_CHECK_EQUAL(size, expected_size);
        UTEST_CHECK_EQUAL(data, expected_data);
    }
}

UTEST_END_MODULE()
