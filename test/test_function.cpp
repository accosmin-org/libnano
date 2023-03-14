#include "fixture/function.h"
#include <nano/function/benchmark/sphere.h>
#include <nano/function/lambda.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_function)

UTEST_CASE(name)
{
    const auto function = function_sphere_t{3};
    UTEST_CHECK_EQUAL(function.name(false), "sphere");
    UTEST_CHECK_EQUAL(function.name(true), "sphere[3D]");
}

UTEST_CASE(lambda)
{
    const auto lambda = [](const vector_t& x, vector_t* gx)
    {
        if (gx != nullptr)
        {
            gx->noalias() = 2 * x;
        }
        return x.dot(x);
    };

    for (tensor_size_t dims = 1; dims < 5; ++dims)
    {
        const auto sphere_function = function_sphere_t{dims};
        const auto lambda_function = make_function(dims, true, true, 2.0, lambda);

        UTEST_CHECK(lambda_function.make(0, 0) == nullptr);

        for (auto trial = 0; trial < 10; ++trial)
        {
            const auto x = make_random_vector<scalar_t>(dims);
            UTEST_CHECK_CLOSE(sphere_function.vgrad(x), lambda_function.vgrad(x), 1e-14);

            auto g1 = make_random_vector<scalar_t>(dims);
            auto g2 = make_random_vector<scalar_t>(dims);
            UTEST_CHECK_CLOSE(sphere_function.vgrad(x, &g1), lambda_function.clone()->vgrad(x, &g2), 1e-14);
            UTEST_CHECK_CLOSE(g1, g2, 1e-14);
        }
    }
}

UTEST_CASE(stats)
{
    for (const auto& function : function_t::make({2, 4, convexity::ignore, smoothness::ignore, 10}))
    {
        UTEST_CHECK_EQUAL(function->fcalls(), 0);
        UTEST_CHECK_EQUAL(function->gcalls(), 0);

        const auto x = make_random_x0(*function);
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

            for (const auto& function : function_t::make({4, 16, convex, smooth, 5}))
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
    for (const auto& rfunction : function_t::make({2, 4, convexity::ignore, smoothness::ignore, 5}))
    {
        const auto& function = *rfunction;
        UTEST_NAMED_CASE(function.name());

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        check_convexity(function);

        UTEST_CHECK_GREATER_EQUAL(function.strong_convexity(), 0.0);
    }
}

UTEST_CASE(grad_accuracy)
{
    for (const auto& rfunction : function_t::make({2, 4, convexity::ignore, smoothness::ignore, 5}))
    {
        const auto& function = *rfunction;
        UTEST_NAMED_CASE(function.name());

        const auto dims = function.size();
        UTEST_CHECK_LESS_EQUAL(dims, 4);
        UTEST_CHECK_GREATER_EQUAL(dims, 2);

        check_gradient(function);
    }
}

UTEST_END_MODULE()
