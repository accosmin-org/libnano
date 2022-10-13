#include <nano/model/util.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_model_util)

UTEST_CASE(sample_with_replacement)
{
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto indices = nano::sample_with_replacement(120, 50);

        UTEST_CHECK_EQUAL(indices.size(), 50);
        UTEST_CHECK_LESS(indices.max(), 120);
        UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
        UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
    }
}

UTEST_CASE(sample_without_replacement)
{
    for (auto trial = 0; trial < 100; ++trial)
    {
        const auto indices = nano::sample_without_replacement(140, 60);

        UTEST_CHECK_EQUAL(indices.size(), 60);
        UTEST_CHECK_LESS(indices.max(), 140);
        UTEST_CHECK_GREATER_EQUAL(indices.min(), 0);
        UTEST_CHECK(std::is_sorted(begin(indices), end(indices)));
        UTEST_CHECK(std::adjacent_find(begin(indices), end(indices)) == end(indices));
    }
}

UTEST_CASE(sample_without_replacement_all)
{
    const auto indices = nano::sample_without_replacement(100, 100);

    UTEST_CHECK_EQUAL(indices, arange(0, 100));
}

UTEST_END_MODULE()
