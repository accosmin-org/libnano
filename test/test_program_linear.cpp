#include <fixture/program.h>
#include <nano/core/strutil.h>

using namespace nano;
using namespace nano::program;

UTEST_BEGIN_MODULE(test_program_linear)

UTEST_CASE(program1)
{
    // see example 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto b = make_vector<scalar_t>(5, 8);

    const auto program = make_linear(c, make_equality(A, b), make_greater(4, 0));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(0.0, 4.0, 1.0, 6.0), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(2.0, 2.0, 1.0, 3.0), 1e-12));

    const auto xbest = make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0);
    check_solution(program, expected_t{xbest}.fbest(xbest.dot(c)));
    check_solution(program, expected_t{xbest}.fbest(xbest.dot(c)));
}

UTEST_CASE(program2)
{
    // see exercise 14.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(1, 0);
    const auto A = make_matrix<scalar_t>(1, 1, 1);
    const auto b = make_vector<scalar_t>(1);

    const auto program = make_linear(c, make_equality(A, b), make_greater(2, 0.0));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(0.0, 1.0), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(1.0, 0.0), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(0.1, 0.9), 1e-12));

    const auto xbest = make_vector<scalar_t>(0.0, 1.0);
    check_solution(program, expected_t{xbest}.fbest(xbest.dot(c)));
}

UTEST_CASE(program3)
{
    // NB: unbounded program!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(1, 0, 1, 1);
    const auto b = make_vector<scalar_t>(2);

    const auto program = make_linear(c, make_equality(A, b), make_greater(3, 0.0));
    check_solution(program, expected_t{}.status(solver_status::unbounded));
}

UTEST_CASE(program4)
{
    // NB: unfeasible program!
    const auto c = make_vector<scalar_t>(-1, 0);
    const auto A = make_matrix<scalar_t>(2, 0, 1, 1, 0);
    const auto b = make_vector<scalar_t>(-1, -1);

    const auto program = make_linear(c, make_equality(A, b), make_greater(2, 0.0));
    check_solution(program, expected_t{}.status(solver_status::unfeasible));
}

UTEST_CASE(program5)
{
    // NB: unfeasible program!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(3, 0, 1, 1, 0, 0, 1, 0, 1, 0);
    const auto b = make_vector<scalar_t>(1, 1, 1);

    const auto program = make_linear(c, make_equality(A, b), make_greater(3, 0.0));
    check_solution(program, expected_t{}.status(solver_status::unfeasible));
}

UTEST_CASE(program_cvx48b)
{
    for (const tensor_size_t dims : {1, 7, 11})
    {
        for (const auto lambda : {-1.0, -1.42, -4.2, -42.1})
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",lambda=", lambda));

            const auto& [program, expected] = make_linear_program_cvx48b(dims, lambda);

            check_solution(program, expected);
        }
    }
}

UTEST_CASE(program_cvx48c)
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

UTEST_END_MODULE()
