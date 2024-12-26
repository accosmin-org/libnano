#include "fixture/function.h"
#include <function/benchmark/sphere.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/lambda.h>
#include <nano/function/util.h>
#include <nano/tensor/stack.h>
#include <unordered_map>

using namespace nano;

namespace
{
scalar_t lambda(const vector_cmap_t x, vector_map_t gx)
{
    if (gx.size() == x.size())
    {
        gx = 2 * x;
    }
    return x.dot(x);
}
} // namespace

UTEST_BEGIN_MODULE(test_function)

UTEST_CASE(name)
{
    const auto function = function_sphere_t{3};
    UTEST_CHECK_EQUAL(function.name(false), "sphere");
    UTEST_CHECK_EQUAL(function.name(true), "sphere[3D]");
}

UTEST_CASE(lambda)
{
    for (tensor_size_t dims = 1; dims < 5; ++dims)
    {
        const auto sphere_function = function_sphere_t{dims};
        const auto lambda_function = make_function(dims, convexity::yes, smoothness::yes, 2.0, lambda);

        UTEST_CHECK(lambda_function.make(0, 0) == nullptr);

        for (auto trial = 0; trial < 10; ++trial)
        {
            const auto x = make_random_vector<scalar_t>(dims);
            UTEST_CHECK_CLOSE(sphere_function(x), lambda_function(x), 1e-14);

            auto g1 = make_random_vector<scalar_t>(dims);
            auto g2 = make_random_vector<scalar_t>(dims);
            UTEST_CHECK_CLOSE(sphere_function(x, g1), lambda_function.clone()->operator()(x, g2), 1e-14);
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
        function->operator()(x);

        UTEST_CHECK_EQUAL(function->fcalls(), 1);
        UTEST_CHECK_EQUAL(function->gcalls(), 0);

        vector_t gx(x.size());
        function->operator()(x, gx);

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
            auto total                 = 0;
            auto counts_per_convexity  = std::unordered_map<bool, int>{};
            auto counts_per_smoothness = std::unordered_map<bool, int>{};
            auto counts_per_size       = std::unordered_map<tensor_size_t, int>{};

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

        check_gradient(function, 100, 1e-8, 1e-14);
    }
}

UTEST_CASE(reduce)
{
    {
        auto A = matrix_t{};
        auto b = vector_t{};
        UTEST_CHECK(!reduce(A, b));
    }

    for (const tensor_size_t dims : {3, 7, 11})
    {
        const auto D = make_random_tensor<scalar_t>(make_dims(2 * dims, dims));
        const auto Q = matrix_t{D.transpose() * D + 0.1 * matrix_t::identity(dims, dims)};
        const auto x = make_random_tensor<scalar_t>(make_dims(dims));

        // full rank
        {
            auto A = matrix_t{Q};
            auto b = vector_t{Q * x};

            const auto expected_A = A;
            const auto expected_b = b;
            UTEST_CHECK(!reduce(A, b));
            UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
            UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
            UTEST_CHECK_CLOSE(vector_t{A * x}, b, 1e-15);
        }

        // duplicated rows
        {
            auto A = ::nano::stack<scalar_t>(2 * dims, dims, Q, Q);
            auto b = ::nano::stack<scalar_t>(2 * dims, Q * x, Q * x);
            UTEST_CHECK(reduce(A, b));
            UTEST_CHECK_EQUAL(A.rows(), dims);
            UTEST_CHECK_EQUAL(b.size(), dims);
            UTEST_CHECK_CLOSE(vector_t{A * x}, b, 1e-14);
        }

        // linear dependency
        {
            auto A = ::nano::stack<scalar_t>(2 * dims, dims, Q, 2.0 * Q);
            auto b = ::nano::stack<scalar_t>(2 * dims, Q * x, 2.0 * (Q * x));
            UTEST_CHECK(reduce(A, b));
            UTEST_CHECK_EQUAL(A.rows(), dims);
            UTEST_CHECK_EQUAL(b.size(), dims);
            UTEST_CHECK_CLOSE(vector_t{A * x}, b, 1e-14);
        }
    }
}

UTEST_CASE(is_convex)
{
    for (const tensor_size_t dims : {3, 7, 11})
    {
        auto Q = matrix_t{matrix_t::identity(dims, dims)};

        UTEST_CHECK(is_convex(Q));
        UTEST_CHECK_CLOSE(strong_convexity(Q), 1.0, epsilon0<scalar_t>());

        UTEST_CHECK(is_convex(2.0 * Q));
        UTEST_CHECK_CLOSE(strong_convexity(Q), 1.0, epsilon0<scalar_t>());

        Q(0, 0) = -1.0;
        UTEST_CHECK(!is_convex(Q));
        UTEST_CHECK_CLOSE(strong_convexity(Q), 0.0, epsilon0<scalar_t>());
    }
}

UTEST_CASE(make_strictly_feasible)
{
    for (const tensor_size_t dims : {3, 7, 11})
    {
        const auto D = make_random_tensor<scalar_t>(make_dims(2 * dims, dims));
        const auto A = matrix_t{D.transpose() * D + 0.1 * matrix_t::identity(dims, dims)};
        const auto x = make_random_tensor<scalar_t>(make_dims(dims));

        for (const auto epsilon : {1e-6, 1e-3, 1e+0})
        {
            const auto b = A * x + epsilon * vector_t::constant(dims, 1.0);

            // feasible: A * z < b
            {
                const auto z = make_strictly_feasible(A, b);
                UTEST_REQUIRE(z.has_value());
                UTEST_CHECK_LESS((A * *z - b).maxCoeff(), 0.0);
            }

            // not feasible: A * z < b and A * z > b + epsilon
            {
                const auto A2 = ::nano::stack<scalar_t>(2 * dims, dims, A, -A);
                const auto b2 = ::nano::stack<scalar_t>(2 * dims, b, -b - vector_t::constant(dims, epsilon));
                const auto z  = make_strictly_feasible(A2, b2);
                UTEST_CHECK(!z.has_value());
            }
        }
    }
}

UTEST_CASE(make_linear_constraints)
{
    auto function = make_function(3, convexity::yes, smoothness::yes, 2.0, lambda);
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(lconstraints.has_value());

        const auto& [A, b, G, h] = lconstraints.value();
        const auto expected_A    = matrix_t{0, 3};
        const auto expected_b    = vector_t{0};
        const auto expected_G    = matrix_t{0, 3};
        const auto expected_h    = vector_t{0};

        UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(G, expected_G, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(h, expected_h, epsilon0<scalar_t>());
    }
    critical(function.variable() >= 2.0);
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(lconstraints.has_value());

        const auto& [A, b, G, h] = lconstraints.value();
        const auto expected_A    = matrix_t{0, 3};
        const auto expected_b    = vector_t{0};
        // clang-format off
        const auto expected_G    = make_tensor<scalar_t>(
            make_dims(3, 3),
            -1, 0, 0,
            0, -1, 0,
            0, 0, -1);
        const auto expected_h    = make_vector<scalar_t>(
            -2,
            -2,
            -2);
        // clang-format on

        UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(G, expected_G, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(h, expected_h, epsilon0<scalar_t>());
    }
    critical(function.variable() <= 3.7);
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(lconstraints.has_value());

        const auto& [A, b, G, h] = lconstraints.value();
        const auto expected_A    = matrix_t{0, 3};
        const auto expected_b    = vector_t{0};
        // clang-format off
        const auto expected_G    = make_tensor<scalar_t>(
            make_dims(6, 3),
            -1, +0, +0,
            +0, -1, +0,
            +0, +0, -1,
            +1, +0, +0,
            +0, +1, +0,
            +0, +0, +1);
        const auto expected_h    = make_vector<scalar_t>(
            -2.0,
            -2.0,
            -2.0,
            +3.7,
            +3.7,
            +3.7);
        // clang-format on

        UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(G, expected_G, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(h, expected_h, epsilon0<scalar_t>());
    }
    critical(vector_t::constant(3, 1.0) * function.variable() == 12.0);
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(lconstraints.has_value());

        const auto& [A, b, G, h] = lconstraints.value();
        // clang-format off
        const auto expected_A    = make_tensor<scalar_t>(
            make_dims(1, 3),
            +1, +1, +1);
        const auto expected_b    = make_vector<scalar_t>(
            12.0);
        const auto expected_G    = make_tensor<scalar_t>(
            make_dims(6, 3),
            -1, +0, +0,
            +0, -1, +0,
            +0, +0, -1,
            +1, +0, +0,
            +0, +1, +0,
            +0, +0, +1);
        const auto expected_h    = make_vector<scalar_t>(
            -2.0,
            -2.0,
            -2.0,
            +3.7,
            +3.7,
            +3.7);
        // clang-format on

        UTEST_CHECK_CLOSE(A, expected_A, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(b, expected_b, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(G, expected_G, epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(h, expected_h, epsilon0<scalar_t>());
    }
    critical(function.constrain(constraint::euclidean_ball_equality_t{make_vector<scalar_t>(0.0, 0.0, 0.0), 30.0}));
    {
        const auto lconstraints = make_linear_constraints(function);
        UTEST_REQUIRE(!lconstraints.has_value());
    }
}

UTEST_END_MODULE()
