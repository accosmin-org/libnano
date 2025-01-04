#include <fixture/program.h>
#include <function/program/numopt162.h>
#include <function/program/numopt1625.h>

using namespace nano;
using namespace constraint;

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
    check_solution(function);
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
    check_solution(function);
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
    check_solution(function);
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
    check_solution(function);
}

UTEST_CASE(program_numopt162)
{
    for (const tensor_size_t dims : {3, 5, 11})
    {
        for (const tensor_size_t neqs : {tensor_size_t{1}, dims - 1, dims})
        {
            const auto function = quadratic_program_numopt162_t{dims, neqs};

            check_convexity(function);
            check_solution(function);
        }
    }
}

UTEST_CASE(program6)
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
    check_solution(function);
}

UTEST_CASE(program7)
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
    check_solution(function);
}

UTEST_CASE(program_numopt1625)
{
    for (const tensor_size_t dims : {2, 3, 7})
    {
        const auto function = quadratic_program_numopt1625_t{dims};

        check_convexity(function);
        check_solution(function);
    }
}

UTEST_END_MODULE()
