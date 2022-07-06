#include "fixture/function.h"
#include <nano/core/numeric.h>
#include <nano/function.h>
#include <nano/function/benchmark.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_function)

UTEST_CASE(stats)
{
    for (const auto& function : benchmark_function_t::make({2, 4, convexity::ignore, smoothness::ignore, 10}))
    {
        UTEST_CHECK_EQUAL(function->fcalls(), 0);
        UTEST_CHECK_EQUAL(function->gcalls(), 0);

        vector_t x = vector_t::Random(function->size());
        function->vgrad(x);

        UTEST_CHECK_EQUAL(function->fcalls(), 1);
        UTEST_CHECK_EQUAL(function->gcalls(), 0);

        vector_t gx(x.size());
        function->vgrad(x, &gx);

        UTEST_CHECK_EQUAL(function->fcalls(), 2);
        UTEST_CHECK_EQUAL(function->gcalls(), 1);

        function->clear_statistics();
        UTEST_CHECK_EQUAL(function->fcalls(), 0);
        UTEST_CHECK_EQUAL(function->gcalls(), 0);
    }
}

UTEST_CASE(select)
{
    for (const auto convex : {convexity::ignore, convexity::yes, convexity::no})
    {
        for (const auto smooth : {smoothness::ignore, smoothness::yes, smoothness::no})
        {
            int                          total = 0;
            std::map<bool, int>          counts_per_convexity;
            std::map<bool, int>          counts_per_smoothness;
            std::map<tensor_size_t, int> counts_per_size;

            for (const auto& function : benchmark_function_t::make({4, 16, convex, smooth, 5}))
            {
                ++total;

                UTEST_CHECK(function != nullptr);
                UTEST_CHECK_LESS_EQUAL(function->size(), 16);
                UTEST_CHECK_GREATER_EQUAL(function->size(), 4);
                UTEST_CHECK(convex == convexity::ignore || function->convex() == (convex == convexity::yes));
                UTEST_CHECK(smooth == smoothness::ignore || function->smooth() == (smooth == smoothness::yes));

                counts_per_size[function->size()]++;
                counts_per_convexity[function->convex()]++;
                counts_per_smoothness[function->smooth()]++;
            }

            UTEST_CHECK_EQUAL(counts_per_size[4], total / 3);
            UTEST_CHECK_EQUAL(counts_per_size[8], total / 3);
            UTEST_CHECK_EQUAL(counts_per_size[16], total / 3);
            UTEST_CHECK_EQUAL(counts_per_convexity[true] + counts_per_convexity[false], total);
            UTEST_CHECK_EQUAL(counts_per_smoothness[true] + counts_per_smoothness[false], total);

            if (convex == convexity::ignore)
            {
                UTEST_CHECK_GREATER(counts_per_convexity[true], 0);
                UTEST_CHECK_GREATER(counts_per_convexity[false], 0);
            }
            else
            {
                UTEST_CHECK_EQUAL(counts_per_convexity[convex != convexity::yes], 0);
            }

            if (smooth == smoothness::ignore)
            {
                UTEST_CHECK_GREATER(counts_per_smoothness[true], 0);
                UTEST_CHECK_GREATER(counts_per_smoothness[false], 0);
            }
            else
            {
                UTEST_CHECK_EQUAL(counts_per_smoothness[smooth != smoothness::yes], 0);
            }
        }
    }
}

UTEST_CASE(convexity)
{
    for (const auto& rfunction : benchmark_function_t::make({2, 4, convexity::ignore, smoothness::ignore, 5}))
    {
        const auto&                 function = *rfunction;
        [[maybe_unused]] const auto _        = utest_test_name_t{function.name()};

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        check_convexity(function);

        UTEST_CHECK_GREATER_EQUAL(function.strong_convexity(), 0.0);
    }
}

UTEST_CASE(grad_accuracy)
{
    for (const auto& rfunction : benchmark_function_t::make({2, 4, convexity::ignore, smoothness::ignore, 5}))
    {
        const auto&                 function = *rfunction;
        [[maybe_unused]] const auto _        = utest_test_name_t{function.name()};

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        check_gradient(function);
    }
}

UTEST_END_MODULE()
