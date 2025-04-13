#include <fixture/solver.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/linear.h>
#include <nano/function/quadratic.h>

using namespace nano;
using namespace constraint;

namespace
{
rsolvers_t make_solvers()
{
    auto solvers = rsolvers_t{};
    for (const auto s0 : {0.99, 0.999, 0.9999})
    {
        for (const auto miu : {5.0, 10.0, 20.0})
        {
            auto solver                            = make_solver("ipm");
            solver->parameter("solver::ipm::s0")   = s0;
            solver->parameter("solver::ipm::miu")  = miu;
            solver->parameter("solver::max_evals") = 100;
            solvers.emplace_back(std::move(solver));
        }
    }
    return solvers;
}
} // namespace

UTEST_BEGIN_MODULE(test_program_quadratic)

UTEST_CASE(constrain)
{
    const auto Q = matrix_t{matrix_t::zero(3, 3)};
    const auto c = vector_t::zero(3);
    const auto a = vector_t::zero(3);
    const auto b = vector_t::zero(2);
    const auto A = matrix_t::zero(2, 3);

    auto function = quadratic_program_t{"qp", Q, c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(A * function.variable() >= b);
    UTEST_REQUIRE(A * function.variable() <= b);
    UTEST_REQUIRE(a * function.variable() == 1.0);
    UTEST_REQUIRE(a * function.variable() >= 1.0);
    UTEST_REQUIRE(a * function.variable() <= 1.0);
    UTEST_REQUIRE(function.variable() >= 1.0);
    UTEST_REQUIRE(function.variable() <= 1.0);
    UTEST_REQUIRE(!function.constrain(functional_equality_t{function}));
    UTEST_REQUIRE(!function.constrain(functional_inequality_t{function}));
    UTEST_REQUIRE(!function.constrain(euclidean_ball_equality_t{vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(euclidean_ball_inequality_t{vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(quadratic_equality_t{matrix_t::zero(3, 3), vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(quadratic_inequality_t{matrix_t::zero(3, 3), vector_t::zero(3), 0.0}));
}

UTEST_CASE(program1)
{
    // see example 16.2, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(6, 2, 1, 5, 2, 4);
    const auto c = make_vector<scalar_t>(-8, -3, -3);
    const auto A = make_matrix<scalar_t>(2, 1, 0, 1, 0, 1, 1);
    const auto b = make_vector<scalar_t>(3, 0);
    const auto x = make_vector<scalar_t>(2, -1, 1);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program2)
{
    // see example p.467, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(0, 2);
    const auto G = -matrix_t::identity(2, 2);
    const auto h = vector_t::zero(2);
    const auto x = make_vector<scalar_t>(0, 0);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program3)
{
    // see example 16.4, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(-2, -5);
    const auto G = make_matrix<scalar_t>(5, -1, 2, 1, 2, 1, -2, -1, 0, 0, -1);
    const auto h = make_vector<scalar_t>(2, 6, 2, 0, 0);
    const auto x = make_vector<scalar_t>(1.4, 1.7);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program4)
{
    // see exercise 16.1a, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(8, 2, 2);
    const auto c = make_vector<scalar_t>(2, 3);
    const auto G = make_matrix<scalar_t>(3, -1, 1, 1, 1, 1, 0);
    const auto h = make_vector<scalar_t>(0, 4, 3);
    const auto x = make_vector<scalar_t>(1.0 / 6.0, -5.0 / 3.0);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program5)
{
    // see exercise 16.11, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, -2, 4);
    const auto c = make_vector<scalar_t>(-2, -6);
    const auto G = make_matrix<scalar_t>(4, 0.5, 0.5, -1, 2, -1, 0, 0, -1);
    const auto h = make_vector<scalar_t>(1, 2, 0, 0);
    const auto x = make_vector<scalar_t>(0.8, 1.2);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program6)
{
    // see exercise 16.17, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(-6, -4);
    const auto G = make_matrix<scalar_t>(3, 1, 1, -1, 0, 0, -1);
    const auto h = make_vector<scalar_t>(3, 0, 0);
    const auto x = make_vector<scalar_t>(2.0, 1.0);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(bundle_cases)
{
    // NB: quadratic programs generated by bundle methods, badly conditioned and hard to solve

    // clang-format off
    const auto Q = make_matrix<scalar_t>(5,
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0);

    const auto c0 = make_vector<scalar_t>(0, 0, 0, 0, 625.0);
    const auto G0 = make_matrix<scalar_t>(3,
        -0.00014353301163777320648, -8.2849464293226782207e-05, 0.00015109222548321000752, 3.7447177848252078335e-05, -1,
        -2.5466140562675764974e-06, -1.4699448434609828959e-06, 2.680732333404427623e-06, 6.6440122998724819222e-07, -1,
        -1.0682720741105252288e-06, -6.1662308934775684817e-07, 1.1245329785614741871e-06, 2.78707830993565414e-07, -1);
    const auto h0 = make_vector<scalar_t>(
        4.8529937564564530991e-06,
        3.2332731948977429066e-09,
        0);

    const auto c1 = make_vector<scalar_t>(0, 0, 0, 0, 6550.5901686479783166);
    const auto G1 = make_matrix<scalar_t>(2,
        9.8983231668534294088e-09, 7.3781561103856015495e-07, -2.457903178239621485e-06, 1.2768656355090551211e-06, -1,
        3.0291065158146719688e-09, 2.2578794783653675608e-07, -7.5217290918316140752e-07, 3.9074921591692585722e-07, -1);
    const auto h1 = make_vector<scalar_t>(
        3.2748492557082926398e-09,
        0);

    const auto c2 = make_vector<scalar_t>(0, 0, 0, 0, 100.0);
    const auto G2 = make_matrix<scalar_t>(2,
        0.012945828710536660261, 0.012945828710536658526, 0.01294582871053666373, 0.0129458287105366672, -1,
        -999999.9926269260468, -999999.9926269260468, -999999.9926269260468, -999999.9926269260468, -1);
    const auto h2 = make_vector<scalar_t>(
        0,
        6.2111205068049457623e-11);
    // clang-format on

    for (const auto& [c, G, h] :
         {std::make_tuple(c0, G0, h0), std::make_tuple(c1, G1, h1), std::make_tuple(c2, G2, h2)})
    {
        static auto index    = 0;
        auto        function = quadratic_program_t{scat("qp-bundle-unscaled-case", index++), Q, c};
        UTEST_REQUIRE(G * function.variable() <= h);

        check_convexity(function);
        check_minimize(make_solvers(), function);
    }
}

UTEST_CASE(factory)
{
    for (const auto& function : function_t::make({2, 16, function_type::quadratic_program}))
    {
        check_convexity(*function);
        check_minimize(make_solvers(), *function);
    }
}

UTEST_END_MODULE()
