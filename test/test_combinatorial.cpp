#include <nano/core/combinatorial.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
template <class tindex, std::enable_if_t<std::is_integral_v<tindex>, bool> = true>
tensor_mem_t<tindex, 2> exhaustive(const tensor_mem_t<tindex, 1>& counts)
{
    auto iter = combinatorial_iterator_t{counts};

    const auto dimensions   = counts.size();
    const auto combinations = iter.size();

    tensor_mem_t<tindex, 2> product(combinations, dimensions);
    for (; iter; ++iter)
    {
        product.tensor(iter.index()) = *iter;
    }

    return product;
}
} // namespace

UTEST_BEGIN_MODULE(test_combinatorial)

UTEST_CASE(exhaustive)
{
    const auto config1 = make_tensor<tensor_size_t>(make_dims(1), 3);
    const auto config2 = make_tensor<tensor_size_t>(make_dims(2), 3, 2);
    const auto config3 = make_tensor<tensor_size_t>(make_dims(3), 3, 2, 2);
    const auto config4 = make_tensor<tensor_size_t>(make_dims(3), 2, 3, 3);

    const auto product1 = make_tensor<tensor_size_t>(make_dims(3, 1), 0, 1, 2);
    const auto product2 = make_tensor<tensor_size_t>(make_dims(6, 2), 0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1);
    const auto product3 = make_tensor<tensor_size_t>(make_dims(12, 3), 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1,
                                                     0, 1, 1, 1, 0, 1, 1, 1, 2, 0, 0, 2, 0, 1, 2, 1, 0, 2, 1, 1);
    const auto product4 = make_tensor<tensor_size_t>(make_dims(18, 3), 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0,
                                                     1, 2, 0, 2, 0, 0, 2, 1, 0, 2, 2, 1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 1,
                                                     0, 1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 2, 1, 1, 2, 2);

    UTEST_CHECK_EQUAL(exhaustive(config1), product1);
    UTEST_CHECK_EQUAL(exhaustive(config2), product2);
    UTEST_CHECK_EQUAL(exhaustive(config3), product3);
    UTEST_CHECK_EQUAL(exhaustive(config4), product4);
}

UTEST_END_MODULE()
