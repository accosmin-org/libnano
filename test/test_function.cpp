#include <utest/utest.h>
#include <nano/function.h>
#include <nano/core/numeric.h>
#include <nano/function/benchmark.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_function)

UTEST_CASE(select)
{
    for (const auto convex : {convexity::ignore, convexity::yes, convexity::no})
    {
        for (const auto smooth : {smoothness::ignore, smoothness::yes, smoothness::no})
        {
            int total = 0;
            std::map<bool, int> counts_per_convexity;
            std::map<bool, int> counts_per_smoothness;
            std::map<tensor_size_t, int> counts_per_size;

            for (const auto& function : benchmark_function_t::make({4, 16, convex, smooth, 5}))
            {
                ++ total;

                UTEST_CHECK(function != nullptr);
                UTEST_CHECK_EQUAL((function->summands() - 1) * (function->summands() - 5), 0);
                UTEST_CHECK_LESS_EQUAL(function->size(), 16);
                UTEST_CHECK_GREATER_EQUAL(function->size(), 4);
                UTEST_CHECK(convex == convexity::ignore || function->convex() == (convex == convexity::yes));
                UTEST_CHECK(smooth == smoothness::ignore || function->smooth() == (smooth == smoothness::yes));

                counts_per_size[function->size()] ++;
                counts_per_convexity[function->convex()] ++;
                counts_per_smoothness[function->smooth()] ++;
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
        const auto& function = *rfunction;
        std::cout << function.name() << std::endl;

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        auto is_convex = true;
        for (auto trial = 0; trial < 100; ++ trial)
        {
            const vector_t x0 = vector_t::Random(dims);
            const vector_t x1 = vector_t::Random(dims);

            is_convex = is_convex && function.is_convex(x0, x1, 20);
        }
        UTEST_CHECK(!function.convex() || is_convex);
    }
}

UTEST_CASE(grad_accuracy)
{
    for (const auto& rfunction : benchmark_function_t::make({2, 4, convexity::ignore, smoothness::ignore, 5}))
    {
        const auto& function = *rfunction;
        std::cout << function.name() << std::endl;

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        for (auto trial = 0; trial < 100; ++ trial)
        {
            const vector_t x = vector_t::Random(dims);
            UTEST_CHECK_LESS(function.grad_accuracy(x), 10 * epsilon2<scalar_t>());
        }
    }
}

UTEST_CASE(summands)
{
    for (const auto& rfunction : benchmark_function_t::make({2, 4, convexity::ignore, smoothness::ignore, 7}))
    {
        const auto& function = *rfunction;
        std::cout << function.name() << std::endl;

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        const auto summands = function.summands();
        if (summands != 1)
        {
            UTEST_CHECK_EQUAL(summands, 7);
        }

        for (auto trial = 0; trial < 100; ++ trial)
        {
            const vector_t x = vector_t::Random(dims);

            vector_t gx = vector_t::Random(dims);
            const auto fx = function.vgrad(x, &gx);

            scalar_t fxsum = 0.0;
            vector_t gxsum = vector_t::Constant(dims, 0.0);
            for (tensor_size_t summand = 0; summand < summands; ++ summand)
            {
                vector_t gxs = vector_t::Random(dims);
                const auto fxs = function.vgrad(x, &gxs, vgrad_config_t{summand});

                fxsum += fxs;
                gxsum += gxs;
            }
            UTEST_CHECK_CLOSE(fx, fxsum / static_cast<scalar_t>(summands), epsilon0<scalar_t>());
            UTEST_CHECK_CLOSE(gx, gxsum / static_cast<scalar_t>(summands), epsilon0<scalar_t>());
        }
    }
}

UTEST_END_MODULE()
