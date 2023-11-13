#include "fixture/program.h"

using namespace nano;
using namespace nano::program;

UTEST_BEGIN_MODULE(test_program_quadratic)

UTEST_CASE(program)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        const auto D = make_random_matrix<scalar_t>(dims, dims);
        const auto c = make_random_vector<scalar_t>(dims);
        {
            const auto Q = matrix_t{matrix_t::zero(dims, dims)};

            const auto program = quadratic_program_t{Q, c};
            UTEST_CHECK(program.convex());
        }
        {
            const auto Q = matrix_t{matrix_t::identity(dims, dims)};

            const auto program = quadratic_program_t{Q, c};
            UTEST_CHECK(program.convex());
        }
        {
            const auto Q = matrix_t{-matrix_t::identity(dims, dims)};

            const auto program = quadratic_program_t{Q, c};
            UTEST_CHECK(!program.convex());
        }
        {
            const auto Q = matrix_t{D.transpose() * D};

            const auto program = quadratic_program_t{Q, c};
            UTEST_CHECK(program.convex());
        }
        {
            const auto Q = matrix_t{D.transpose() * D + matrix_t::identity(dims, dims)};

            const auto program = quadratic_program_t{Q, c};
            UTEST_CHECK(program.convex());
        }
        {
            const auto Q = matrix_t{-D.transpose() * D - matrix_t::identity(dims, dims)};

            const auto program = quadratic_program_t{Q, c};
            UTEST_CHECK(!program.convex());
        }
        {
            auto Q = matrix_t{matrix_t::identity(dims, dims)};
            Q(0, 1) += 1.0;

            const auto program = quadratic_program_t{Q, c};
            UTEST_CHECK(!program.convex());
        }
    }
}

UTEST_CASE(program1)
{
    // see example 16.2, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(6, 2, 1, 5, 2, 4);
    const auto c = make_vector<scalar_t>(-8, -3, -3);
    const auto A = make_matrix<scalar_t>(2, 1, 0, 1, 0, 1, 1);
    const auto b = make_vector<scalar_t>(3, 0);
    const auto Q = make_matrix<scalar_t>(3, 6, 2, 1, 2, 5, 2, 1, 2, 4);

    const auto program = make_quadratic_upper_triangular(q, c, make_equality(A, b));
    UTEST_CHECK(program.convex());
    UTEST_CHECK_CLOSE(program.m_Q, Q, 1e-15);
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(1, -2, 2), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(2, -1, 1), 1e-12));
    UTEST_CHECK(!program.feasible(make_vector<scalar_t>(1, 1, 1), 1e-12));
    UTEST_CHECK(!program.feasible(make_vector<scalar_t>(1, 1, 2), 1e-12));

    const auto xbest = make_vector<scalar_t>(2, -1, 1);
    check_solution(program, expected_t{xbest}.fbest(-3.5));
}

UTEST_CASE(program2)
{
    // see example p.467, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(0, 2);
    const auto G = -matrix_t::identity(2, 2);
    const auto h = vector_t::zero(2);
    const auto Q = make_matrix<scalar_t>(2, 2, 0, 0, 2);

    const auto program = make_quadratic_upper_triangular(q, c, make_inequality(G, h));
    UTEST_CHECK(program.convex());
    UTEST_CHECK_CLOSE(program.m_Q, Q, 1e-15);
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(1, 1), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(1, 0), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(0, 0), 1e-12));
    UTEST_CHECK(!program.feasible(make_vector<scalar_t>(-1, 1), 1e-12));
    UTEST_CHECK(!program.feasible(make_vector<scalar_t>(1, -1), 1e-12));

    const auto xbest = make_vector<scalar_t>(0, 0);
    check_solution(program, expected_t{xbest}.fbest(0));
}

UTEST_CASE(program3)
{
    // see example 16.4, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(-2, -5);
    const auto G = make_matrix<scalar_t>(5, -1, 2, 1, 2, 1, -2, -1, 0, 0, -1);
    const auto h = make_vector<scalar_t>(2, 6, 2, 0, 0);
    const auto Q = make_matrix<scalar_t>(2, 2, 0, 0, 2);

    const auto program = make_quadratic_upper_triangular(q, c, make_inequality(G, h));
    UTEST_CHECK(program.convex());
    UTEST_CHECK_CLOSE(program.m_Q, Q, 1e-15);
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(1, 1), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(1, 0), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(0, 0), 1e-12));

    const auto xbest = make_vector<scalar_t>(1.4, 1.7);
    check_solution(program, expected_t{xbest}.fbest(-6.45));
}

UTEST_CASE(program4)
{
    // see exercise 16.1a, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(8, 2, 2);
    const auto c = make_vector<scalar_t>(2, 3);
    const auto G = make_matrix<scalar_t>(3, -1, 1, 1, 1, 1, 0);
    const auto h = make_vector<scalar_t>(0, 4, 3);
    const auto Q = make_matrix<scalar_t>(2, 8, 2, 2, 2);

    const auto program = make_quadratic_upper_triangular(q, c, make_inequality(G, h));
    UTEST_CHECK(program.convex());
    UTEST_CHECK_CLOSE(program.m_Q, Q, 1e-15);
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(1, 1), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(1, 0), 1e-12));
    UTEST_CHECK(program.feasible(make_vector<scalar_t>(0, 0), 1e-12));
    UTEST_CHECK(!program.feasible(make_vector<scalar_t>(0, 1), 1e-12));

    const auto xbest = make_vector<scalar_t>(1.0 / 6.0, -5.0 / 3.0);
    check_solution(program, expected_t{xbest}.fbest(-7.0 / 3.0));
}

UTEST_CASE(program5)
{
    // see exercise 16.2, "Numerical optimization", Nocedal & Wright, 2nd edition
    for (const tensor_size_t dims : {3, 5, 11})
    {
        const auto x0 = make_random_vector<scalar_t>(dims);
        const auto Q  = matrix_t{matrix_t::identity(dims, dims)};
        const auto c  = vector_t{-x0};

        for (const tensor_size_t neqs : {tensor_size_t{1}, dims - 1, dims})
        {
            auto L = make_random_matrix<scalar_t>(neqs, neqs);
            auto U = make_random_matrix<scalar_t>(neqs, dims);

            L.matrix().triangularView<Eigen::Upper>().setZero();
            U.matrix().triangularView<Eigen::Lower>().setZero();

            L.diagonal().array() = 1.0;
            U.diagonal().array() = 1.0;

            const auto A = L * U;
            const auto b = make_random_vector<scalar_t>(neqs);

            const auto program = make_quadratic(Q, c, make_equality(A, b));
            UTEST_CHECK(program.convex());

            const auto invAA = (A * A.transpose()).inverse();
            const auto xbest = vector_t{x0 + A.transpose() * invAA * (b - A * x0)};
            const auto vbest = vector_t{-invAA * (b - A * x0)};
            const auto fbest = scalar_t{0.5 * (b - A * x0).dot(invAA * (b - A * x0))} - 0.5 * x0.dot(x0);
            const auto state = check_solution(program, expected_t{xbest}.fbest(fbest));
            UTEST_CHECK_CLOSE(state.m_v, vbest, 1e-10);
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
    const auto Q = make_matrix<scalar_t>(2, 2, -2, -2, 4);

    const auto program = make_quadratic_upper_triangular(q, c, make_inequality(G, h));
    UTEST_CHECK(program.convex());
    UTEST_CHECK_CLOSE(program.m_Q, Q, 1e-15);

    const auto xbest = make_vector<scalar_t>(0.8, 1.2);
    check_solution(program, expected_t{xbest}.fbest(-7.2));
    check_solution(program, expected_t{xbest}.x0(make_vector<scalar_t>(0.1, 0.2)).fbest(-7.2));
    check_solution(program, expected_t{xbest}.x0(make_vector<scalar_t>(0.2, 0.1)).fbest(-7.2));
    check_solution(program, expected_t{xbest}.x0(make_vector<scalar_t>(0.0, 0.0)).status(solver_status::unfeasible));
    check_solution(program, expected_t{xbest}.x0(make_vector<scalar_t>(-0.1, -0.3)).status(solver_status::unfeasible));
}

UTEST_CASE(program7)
{
    // see exercise 16.17, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(-6, -4);
    const auto G = make_matrix<scalar_t>(3, 1, 1, -1, 0, 0, -1);
    const auto h = make_vector<scalar_t>(3, 0, 0);
    const auto Q = make_matrix<scalar_t>(2, 2, 0, 0, 2);

    const auto program = make_quadratic_upper_triangular(q, c, make_inequality(G, h));
    UTEST_CHECK(program.convex());
    UTEST_CHECK_CLOSE(program.m_Q, Q, 1e-15);

    const auto xbest = make_vector<scalar_t>(2.0, 1.0);
    check_solution(program, expected_t{xbest}.fbest(-11));
}

UTEST_CASE(program8)
{
    // see exercise 16.25, "Numerical optimization", Nocedal & Wright, 2nd edition
    for (const tensor_size_t dims : {2, 3, 7})
    {
        UTEST_NAMED_CASE(scat("dims=", dims));

        const auto x0 = make_random_vector<scalar_t>(dims);
        const auto Q  = matrix_t{matrix_t::identity(dims, dims)};
        const auto c  = vector_t{-x0};
        const auto l  = make_random_vector<scalar_t>(dims);
        const auto u  = vector_t{l.array() + 0.1};

        const auto program = make_quadratic(Q, c, make_greater(l), make_less(u));
        UTEST_CHECK(program.convex());

        const auto xbest = vector_t{x0.array().max(l.array()).min(u.array())};
        const auto fbest = 0.5 * xbest.dot(xbest) - xbest.dot(x0);
        check_solution(program, expected_t{xbest}.fbest(fbest));
    }
}

UTEST_END_MODULE()
