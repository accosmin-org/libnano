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
    for (const auto s0 : {0.99, 0.999})
    {
        for (const auto miu : {5.0, 10.0, 20.0})
        {
            for (const auto gamma : {2.0, 3.0})
            {
                auto solver                             = make_solver("ipm");
                solver->parameter("solver::ipm::s0")    = s0;
                solver->parameter("solver::ipm::miu")   = miu;
                solver->parameter("solver::ipm::gamma") = gamma;
                solver->parameter("solver::max_evals")  = 100;
                solvers.emplace_back(std::move(solver));
            }
        }
    }
    return solvers;
}
} // namespace

UTEST_BEGIN_MODULE(test_program_linear)

UTEST_CASE(constrain)
{
    const auto c = vector_t::zero(3);
    const auto a = vector_t::zero(3);
    const auto b = vector_t::zero(2);
    const auto A = matrix_t::zero(2, 3);

    auto function = linear_program_t{"lp", c};
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
    // see example 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto b = make_vector<scalar_t>(5, 8);
    const auto x = make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0);

    auto function = linear_program_t{"lp1", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program2)
{
    // see exercise 14.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(1, 0);
    const auto A = make_matrix<scalar_t>(1, 1, 1);
    const auto b = make_vector<scalar_t>(1);
    const auto x = make_vector<scalar_t>(0.0, 1.0);

    auto function = linear_program_t{"lp2", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(factory)
{
    for (const auto& function : function_t::make({2, 16, function_type::linear_program}))
    {
        check_convexity(*function);
        check_minimize(make_solvers(), *function);
    }
}

UTEST_CASE(regression1)
{
    const auto function = make_function("cvx48e-ineq", "function::seed", 3991, "function::cvx48e-ineq::alpha", 1e-6);

    const auto x0 =
        make_vector<scalar_t>(-0.2764936187966942, -0.2222212963715804, 0.09466578040479257, 0.5659487082254151,
                              0.6548357325973591, -0.2565754309721414, -0.7536298826268002, 0.8886021436691338,
                              -0.9769280576612476, 0.6916720138178716, -0.4352138222311691, -0.08738905703404498,
                              0.09437059025466077, -0.6509127504009076, 0.5330721389348336, -0.8400094985342113);

    check_minimize(make_solvers(), *(function->make(16)), x0);
}

UTEST_CASE(regression2)
{
    const auto function = make_function("cvx48e-ineq", "function::seed", 6512, "function::cvx48e-ineq::alpha", 1e-6);

    const auto x0 = make_vector<scalar_t>(-0.962132710072084, -0.4827775792380838, 0.5287349089933873);

    check_minimize(make_solvers(), *(function->make(3)), x0);
}

UTEST_CASE(regression3)
{
    const auto function = make_function("cvx48e-ineq", "function::seed", 6512, "function::cvx48e-ineq::alpha", 1e-6);

    const auto x0 = make_vector<scalar_t>(0.9869452885804215, -0.7095886246222366, -0.7581925416733991);

    check_minimize(make_solvers(), *(function->make(3)), x0);
}

UTEST_END_MODULE()
