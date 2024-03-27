#include "fixture/program.h"

#include <Eigen/Dense>

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

            const auto muv   = std::max({1.0, A.lpNorm<2>(), b.lpNorm<2>()});
            const auto mux   = std::max({1.0, Q.lpNorm<2>(), c.lpNorm<2>()});
            const auto invAA = (A * A.transpose()).inverse();
            const auto xbest = vector_t{x0 + A.transpose() * invAA * (b - A * x0)};
            const auto vbest = vector_t{-invAA * (b - A * x0) * muv / mux};
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

UTEST_CASE(program9)
{
    // badly scaled programs generated with the RQB solver applied to linear machine learning problems.
    const auto Q1 = make_matrix<scalar_t>(
        6, 7695057.3606177885085344, -7692711.7498994730412960, 1774665.9566367159131914, -2958099.6455304687842727,
        593055.4774447004310787, -2957389.7971845343708992, -7692711.7498994730412960, 7690370.3438775558024645,
        -1778501.9468738515861332, 2956050.9844734095968306, -592876.0527072392869741, 2957489.0522283604368567,
        1774665.9566367159131914, -1778501.9468738515861332, 7688594.0792828639969230, -1777899.3335352085996419,
        -593608.1158854841487482, -1777189.4851892746519297, -2958099.6455304687842727, 2956050.9844734095968306,
        -1777899.3335352085996419, 7690518.4962502717971802, -2959509.8127272250130773, 590636.6286198728485033,
        593055.4774446999654174, -592876.0527072392869741, -593608.1158854841487482, -2959509.8127272245474160,
        7692237.0262242779135704, -2958799.9643812905997038, -2957389.7971845343708992, 2957489.0522283604368567,
        -1777189.4851892746519297, 590636.6286198727320880, -2958799.9643812905997038, 7691938.1929421387612820);

    const auto c1 = make_vector<scalar_t>(0.0000000000000000, 286.0212216630087028, 0.0000396148702730,
                                          0.0000951540357619, 0.0000492518259509, 0.0000961890000983);

    const auto Q2 =
        make_matrix<scalar_t>(3, 769254010.1276453733444214, -769258932.4067106246948242, -59174331.5974321961402893,
                              -769258932.4067106246948242, 769263856.1250183582305908, 59151610.9445311576128006,
                              -59174331.5974321961402893, 59151610.9445311576128006, 769202060.4053010940551758);

    const auto c2 = make_vector<scalar_t>(0.0000000000000000, 8886.7208660855503695, 0.0000032102354108);

    const auto Q3 = make_matrix<scalar_t>(
        3, 7692308262809.2568359375000000, -7692310225375.0507812500000000, 591714357016.8245849609375000,
        -7692310225375.0507812500000000, 7692312187943.5097656250000000, -591717753629.4758300781250000,
        591714357016.8245849609375000, -591717753629.4758300781250000, 7692303883177.0546875000000000);

    const auto c3 = make_vector<scalar_t>(0.0000000000000000, 5588.7619455829144499, 0.0033108046837427);

    for (const auto& [Q, c] : {std::make_tuple(Q1, c1), std::make_tuple(Q2, c2), std::make_tuple(Q3, c3)})
    {
        UTEST_NAMED_CASE(scat("c=", c));

        const auto dims  = c.size();
        const auto lower = program::make_less(dims, 1.0);
        const auto upper = program::make_greater(dims, 0.0);
        const auto wsum1 = program::make_equality(vector_t::constant(dims, 1.0), 1.0);

        const auto program = program::make_quadratic(Q, c, lower, upper, wsum1);
        UTEST_CHECK(program.convex());

        const auto x0 = vector_t{vector_t::constant(dims, 1.0 / static_cast<scalar_t>(dims))};
        assert(program.feasible(x0, epsilon1<scalar_t>()));

        check_solution_(program, expected_t{}.x0(x0));
    }
}

UTEST_END_MODULE()
