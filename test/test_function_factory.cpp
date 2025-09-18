#include <fixture/function.h>
#include <function/nonlinear/sphere.h>
#include <nano/function/lambda.h>
#include <unordered_map>

using namespace nano;

namespace
{
scalar_t lambda(const vector_cmap_t x, vector_map_t gx, matrix_map_t Hx)
{
    if (gx.size() == x.size())
    {
        gx = x;
    }

    if (Hx.rows() == x.size() && Hx.cols() == x.size())
    {
        Hx = matrix_t::identity(x.size(), x.size());
    }

    return 0.5 * x.dot(x);
}
} // namespace

UTEST_BEGIN_MODULE()

UTEST_CASE(lambda)
{
    for (tensor_size_t dims = 1; dims < 5; ++dims)
    {
        const auto sphere_function = function_sphere_t{dims};
        const auto lambda_function = make_function(dims, convexity::yes, smoothness::yes, 2.0, lambda);

        UTEST_CHECK(lambda_function.make(0) == nullptr);

        for (auto trial = 0; trial < 10; ++trial)
        {
            const auto x = make_random_vector<scalar_t>(dims);
            UTEST_CHECK_CLOSE(sphere_function(x), lambda_function(x), 1e-14);

            auto g1 = make_random_vector<scalar_t>(dims);
            auto g2 = make_random_vector<scalar_t>(dims);
            auto h1 = make_random_matrix<scalar_t>(dims, dims);
            auto h2 = make_random_matrix<scalar_t>(dims, dims);
            UTEST_CHECK_CLOSE(sphere_function(x, g1, h1), lambda_function.clone()->operator()(x, g2, h2), 1e-14);
            UTEST_CHECK_CLOSE(g1, g2, 1e-14);
            UTEST_CHECK_CLOSE(h1, h2, 1e-14);
        }
    }
}

UTEST_CASE(stats)
{
    for (const auto& function : function_t::make({2, 4, function_type::any}))
    {
        UTEST_NAMED_CASE(function->name());

        UTEST_CHECK_EQUAL(function->fcalls(), 0);
        UTEST_CHECK_EQUAL(function->gcalls(), 0);
        UTEST_CHECK_EQUAL(function->hcalls(), 0);

        const auto x  = make_random_x0(*function);
        auto       gx = vector_t{x.size()};
        auto       Hx = matrix_t{x.size(), x.size()};

        function->operator()(x);
        UTEST_CHECK_EQUAL(function->fcalls(), 1);
        UTEST_CHECK_EQUAL(function->gcalls(), 0);
        UTEST_CHECK_EQUAL(function->hcalls(), 0);

        function->operator()(x, gx);
        UTEST_CHECK_EQUAL(function->fcalls(), 2);
        UTEST_CHECK_EQUAL(function->gcalls(), 1);
        UTEST_CHECK_EQUAL(function->hcalls(), 0);

        if (function->smooth())
        {
            function->operator()(x, gx, Hx);
            UTEST_CHECK_EQUAL(function->fcalls(), 3);
            UTEST_CHECK_EQUAL(function->gcalls(), 2);
            UTEST_CHECK_EQUAL(function->hcalls(), 1);
        }

        function->clear_statistics();
        UTEST_CHECK_EQUAL(function->fcalls(), 0);
        UTEST_CHECK_EQUAL(function->gcalls(), 0);
        UTEST_CHECK_EQUAL(function->hcalls(), 0);
    }
}

UTEST_CASE(select)
{
    // clang-format off
    const auto types =
    {
        function_type::convex,
        function_type::smooth,
        function_type::convex_smooth,
        function_type::convex_nonsmooth
    };
    // clang-format on

    for (const auto fun_type : types)
    {
        UTEST_NAMED_CASE(scat(fun_type));

        auto total                 = 0;
        auto counts_per_convexity  = std::unordered_map<bool, int>{};
        auto counts_per_smoothness = std::unordered_map<bool, int>{};
        auto counts_per_size       = std::unordered_map<tensor_size_t, int>{};

        // clang-format off
        const auto expects_convex =
            (fun_type == function_type::convex) ||
            (fun_type == function_type::convex_smooth) ||
            (fun_type == function_type::convex_nonsmooth);

        const auto expects_smooth =
            (fun_type == function_type::smooth) ||
            (fun_type == function_type::convex_smooth);
        // clang-format on

        for (const auto& function : function_t::make({4, 16, fun_type}))
        {
            ++total;

            UTEST_CHECK(function != nullptr);
            UTEST_CHECK_LESS_EQUAL(function->size(), 16);
            UTEST_CHECK_GREATER_EQUAL(function->size(), 4);
            UTEST_CHECK(!expects_convex || function->convex());
            UTEST_CHECK(!expects_smooth || function->smooth());

            counts_per_size[function->size()]++;
            counts_per_convexity[function->convex()]++;
            counts_per_smoothness[function->smooth()]++;
        }

        UTEST_CHECK_EQUAL(counts_per_size[4], total / 3);
        UTEST_CHECK_EQUAL(counts_per_size[8], total / 3);
        UTEST_CHECK_EQUAL(counts_per_size[16], total / 3);

        if (expects_convex)
        {
            UTEST_CHECK_GREATER(counts_per_convexity[true], 0);
            UTEST_CHECK_EQUAL(counts_per_convexity[false], 0);
        }
        else
        {
            UTEST_CHECK_GREATER(counts_per_convexity[true], 0);
            UTEST_CHECK_GREATER(counts_per_convexity[false], 0);
        }

        if (expects_smooth)
        {
            UTEST_CHECK_GREATER(counts_per_smoothness[true], 0);
            UTEST_CHECK_EQUAL(counts_per_smoothness[false], 0);
        }
        else if (fun_type == function_type::convex_nonsmooth)
        {
            UTEST_CHECK_EQUAL(counts_per_smoothness[true], 0);
            UTEST_CHECK_GREATER(counts_per_smoothness[false], 0);
        }
        else
        {
            UTEST_CHECK_GREATER(counts_per_smoothness[true], 0);
            UTEST_CHECK_GREATER(counts_per_smoothness[false], 0);
        }
    }
}

UTEST_CASE(reproducibility)
{
    for (const auto& rfunction : function_t::make({2, 16, function_type::any}))
    {
        auto& function = *rfunction;
        UTEST_NAMED_CASE(function.name());

        auto rfunctions = rfunctions_t{};
        if (function.parameter_if("function::seed") != nullptr)
        {
            const auto seed0 = function.parameter("function::seed").value<uint64_t>();

            for (const auto seed : {seed0, seed0 + 1, seed0 + 87, seed0 + 347, seed0 + 1786})
            {
                function.parameter("function::seed") = seed % 10001;
                rfunctions.emplace_back(function.make(function.size()));
                rfunctions.emplace_back(function.make(function.size()));
            }
        }
        else
        {
            rfunctions.emplace_back(function.clone());
        }

        for (tensor_size_t trial = 0, trials = 5; trial < trials; ++trial)
        {
            const auto x      = make_random_vector<scalar_t>(function.size());
            const auto nseeds = static_cast<tensor_size_t>(rfunctions.size());

            auto fxs = make_random_vector<scalar_t>(nseeds);
            auto gxs = make_random_matrix<scalar_t>(nseeds, function.size());

            // should obtain the same output for the same random input and the same random seed
            for (tensor_size_t i = 0; i < nseeds; ++i)
            {
                const auto& seed_function = *(rfunctions[static_cast<size_t>(i)]);

                auto gx = make_random_vector<scalar_t>(function.size());

                fxs(i)        = seed_function(x, gxs.tensor(i));
                const auto fx = seed_function(x, gx);

                const auto df = std::fabs(fxs(i) - fx);
                const auto dg = (gxs.tensor(i) - gx).lpNorm<Eigen::Infinity>();

                UTEST_CHECK_LESS(df, epsilon0<scalar_t>());
                UTEST_CHECK_LESS(dg, epsilon0<scalar_t>());
            }

            // check reproducibility of outputs across random seeds
            for (tensor_size_t i = 0; i + 2 < nseeds; i += 2)
            {
                // same seed => same outputs
                {
                    const auto df = std::fabs(fxs(i) - fxs(i + 1));
                    const auto dg = (gxs.tensor(i) - gxs.tensor(i + 1)).lpNorm<Eigen::Infinity>();

                    UTEST_CHECK_LESS(df, epsilon0<scalar_t>());
                    UTEST_CHECK_LESS(dg, epsilon0<scalar_t>());
                }

                // different seeds => different outputs
                // NB: ignore discontinuous functions as it is very likely for low dimensions to produce similar
                //     function values and gradients even for different seeds!
                if (nano::starts_with(function.name(), "kinks") || ///<
                    nano::starts_with(function.name(), "mae") ||   ///<
                    nano::starts_with(function.name(), "hinge"))   ///<
                {
                    continue;
                }
                for (tensor_size_t j = i + 2; j < nseeds; ++j)
                {
                    const auto df = std::fabs(fxs(i) - fxs(j));
                    const auto dg = (gxs.tensor(i) - gxs.tensor(j)).lpNorm<Eigen::Infinity>();

                    UTEST_CHECK_GREATER(df, 1e+2 * epsilon0<scalar_t>());
                    UTEST_CHECK_GREATER(dg, epsilon1<scalar_t>());
                }
            }
        }
    }
}

UTEST_END_MODULE()
