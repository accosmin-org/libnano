#include <fixture/program.h>
#include <function/program/cvx48b.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_program_linear)

UTEST_CASE(program1)
{
    // see example 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto b = make_vector<scalar_t>(5, 8);
    const auto x = make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0);

    auto function = linear_program_t{"lp", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(x));

    check_solution(function);
}

UTEST_CASE(program2)
{
    // see exercise 14.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(1, 0);
    const auto A = make_matrix<scalar_t>(1, 1, 1);
    const auto b = make_vector<scalar_t>(1);
    const auto x = make_vector<scalar_t>(0.0, 1.0);

    auto function = linear_program_t{"lp", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(x));

    check_solution(function);
}

UTEST_CASE(program3)
{
    // NB: unbounded program!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(1, 0, 1, 1);
    const auto b = make_vector<scalar_t>(2);

    auto function = linear_program_t{"lp", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(vector_t{}, solver_status::unbounded));

    check_solution(function);
}

UTEST_CASE(program4)
{
    // NB: unfeasible program!
    const auto c = make_vector<scalar_t>(-1, 0);
    const auto A = make_matrix<scalar_t>(2, 0, 1, 1, 0);
    const auto b = make_vector<scalar_t>(-1, -1);

    auto function = linear_program_t{"lp", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(vector_t{}, solver_status::unfeasible));

    check_solution(function);
}

UTEST_CASE(program5)
{
    // NB: unfeasible program!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(3, 0, 1, 1, 0, 0, 1, 0, 1, 0);
    const auto b = make_vector<scalar_t>(1, 1, 1);

    auto function = linear_program_t{"lp", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(vector_t{}, solver_status::unfeasible));

    check_solution(function);
}

UTEST_CASE(program_cvx48b)
{
    for (const tensor_size_t dims : {1, 7, 11})
    {
        for (const auto lambda : {-1.0, -1.42, -4.2, -42.1})
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",lambda=", lambda));

            const auto function = linear_program_cvx48b_t{dims, lambda};

            check_solution(function);
        }
    }
}

/*UTEST_CASE(program_cvx48c)
{
    for (const tensor_size_t dims : {1, 7, 11})
    {
        UTEST_NAMED_CASE(scat("dims=", dims));

        const auto& [program, expected] = make_linear_program_cvx48c(dims);

        check_solution(program, expected);
    }
}

UTEST_CASE(program_cvx48d_eq)
{
    for (const tensor_size_t dims : {2, 4, 9})
    {
        UTEST_NAMED_CASE(scat("dims=", dims, ",x.sum()==1"));

        const auto& [program, expected] = make_linear_program_cvx48d_eq(dims);

        check_solution(program, expected);
    }
}

UTEST_CASE(program_cvx48d_ineq)
{
    for (const tensor_size_t dims : {2, 5, 8})
    {
        UTEST_NAMED_CASE(scat("dims=", dims, ",x.sum()<=1"));

        const auto& [program, expected] = make_linear_program_cvx48d_ineq(dims);

        check_solution(program, expected);
    }
}

UTEST_CASE(program_cvx48e_eq)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (tensor_size_t alpha = 0; alpha <= dims; ++alpha)
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",alpha=", alpha, ",x.sum()==alpha"));

            const auto& [program, expected] = make_linear_program_cvx48e_eq(dims, alpha);

            check_solution(program, expected);
        }
    }
}

UTEST_CASE(program_cvx48e_ineq)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (tensor_size_t alpha = 0; alpha <= dims; ++alpha)
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",alpha=", alpha, ",x.sum()<=alpha"));

            const auto& [program, expected] = make_linear_program_cvx48e_ineq(dims, alpha);

            check_solution(program, expected);
        }
    }
}

UTEST_CASE(program_cvx48f)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (const auto alpha : {0.0, 0.3, 0.7, 1.0})
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",alpha=", alpha));

            const auto& [program, expected] = make_linear_program_cvx48f(dims, alpha);

            check_solution(program, expected);
        }
    }
}

UTEST_CASE(program_cvx49)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        UTEST_NAMED_CASE(scat("dims=", dims));

        const auto& [program, expected] = make_linear_program_cvx49(dims);

        check_solution(program, expected);
    }
}

UTEST_CASE(program_cvx410)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        UTEST_NAMED_CASE(scat("feasible(dims=", dims, ")"));

        const auto feasible             = true;
        const auto& [program, expected] = make_linear_program_cvx410(dims, feasible);

        check_solution(program, expected);
    }
}

UTEST_CASE(program_cvx410_unfeasible)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        UTEST_NAMED_CASE(scat("unfeasible(dims=", dims, ")"));

        const auto feasible             = false;
        const auto& [program, expected] = make_linear_program_cvx410(dims, feasible);

        check_solution(program, expected);
    }
}
*/

UTEST_END_MODULE()
