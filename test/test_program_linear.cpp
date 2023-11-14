#include "fixture/program.h"
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
    check_solution(program, expected_t{xbest}.fbest(xbest.dot(c)).use_logger(false));
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
    check_solution(program, expected_t{}.status(solver_status::stopped));
    // FIXME: check_solution(program, expected_t{}.status(solver_status::unbounded));
}

UTEST_CASE(program4)
{
    // NB: unfeasible program!
    const auto c = make_vector<scalar_t>(-1, 0);
    const auto A = make_matrix<scalar_t>(2, 0, 1, 1, 0);
    const auto b = make_vector<scalar_t>(-1, -1);

    const auto program = make_linear(c, make_equality(A, b), make_greater(2, 0.0));
    check_solution(program, expected_t{}.status(solver_status::stopped));
    // FIXME: check_solution(program, expected_t{}.status(solver_status::unbounded));
}

UTEST_CASE(program5)
{
    // NB: unfeasible program!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(3, 0, 1, 1, 0, 0, 1, 0, 1, 0);
    const auto b = make_vector<scalar_t>(1, 1, 1);

    const auto program = make_linear(c, make_equality(A, b), make_greater(3, 0.0));
    check_solution(program, expected_t{}.status(solver_status::stopped));
    // FIXME: check_solution(program, expected_t{}.status(solver_status::unbounded));
}

UTEST_CASE(program6)
{
    // exercise 4.8 (b), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over a halfspace:
    //  min c.dot(x) s.t. a.dot(x) <= b,
    //  where c = lambda * a.
    for (const tensor_size_t dims : {1, 7, 11})
    {
        for (const auto lambda : {-1.0, -1.42, -4.2, -42.1})
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",lambda=", lambda));

            const auto a = make_random_vector<scalar_t>(dims, +1.0, +2.0);
            const auto b = urand<scalar_t>(-1.0, +1.0);
            const auto c = lambda * a;

            const auto program  = make_linear(c, make_inequality(a, b));
            const auto solution = check_solution(program, expected_t{});

            const auto fbest = lambda * b;
            UTEST_CHECK_CLOSE(solution.m_fx, fbest, 1e-12);
            UTEST_CHECK_CLOSE(solution.m_x.dot(c), fbest, 1e-12);
            UTEST_CHECK_CLOSE(solution.m_x.dot(a), b, 1e-12);
        }
    }
}

UTEST_CASE(program7)
{
    // exercise 4.8 (c), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over a rectangle:
    //  min c.dot(x) s.t. l <= x <= u,
    //  where l <= u.
    for (const tensor_size_t dims : {1, 7, 11})
    {
        UTEST_NAMED_CASE(scat("dims=", dims));

        const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
        const auto l = make_random_vector<scalar_t>(dims, -1.0, +1.0);
        const auto u = make_random_vector<scalar_t>(dims, +1.0, +3.0);

        const auto program  = make_linear(c, make_greater(l), make_less(u));
        const auto xbest    = vector_t{l.array() * c.array().max(0.0).sign() - u.array() * c.array().min(0.0).sign()};
        const auto solution = check_solution(program, expected_t{xbest});
        UTEST_CHECK_GREATER_EQUAL((solution.m_x - l).minCoeff(), -1e-12);
        UTEST_CHECK_GREATER_EQUAL((u - solution.m_x).minCoeff(), -1e-12);
    }
}

UTEST_CASE(program8)
{
    const auto make_xbest = [&](const vector_t& c)
    {
        const auto dims = c.size();
        const auto cmin = c.min();

        auto count = 0.0;
        auto xbest = make_full_vector<scalar_t>(dims, 0.0);
        for (tensor_size_t i = 0; i < dims; ++i)
        {
            if (c(i) == cmin)
            {
                ++count;
                xbest(i) = 1.0;
            }
        }
        xbest.array() /= count;
        return xbest;
    };

    // exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over the probability simplex:
    //  min c.dot(x) s.t. 1.dot(x) = 1, x >= 0.
    for (const tensor_size_t dims : {2, 4, 9})
    {
        UTEST_NAMED_CASE(scat("dims=", dims, ",x.sum()==1"));

        const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
        const auto A = vector_t::constant(dims, 1.0);
        const auto b = 1.0;

        const auto program = make_linear(c, make_equality(A, b), make_greater(dims, 0.0));
        const auto xbest   = make_xbest(c);
        check_solution(program, expected_t{xbest});
    }

    // exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over the probability simplex:
    //  min c.dot(x) s.t. 1.dot(x) <= 1, x >= 0.
    for (const tensor_size_t dims : {2, 5, 8})
    {
        UTEST_NAMED_CASE(scat("dims=", dims, ",x.sum()<=1"));

        const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
        const auto A = vector_t::constant(dims, 1.0);
        const auto N = -matrix_t::identity(dims, dims);
        const auto b = 1.0;
        const auto z = vector_t::constant(dims, 0.0);

        const auto program = make_linear(c, make_inequality(A, b), make_inequality(N, z), make_greater(dims, 0.0));
        const auto xbest   = c.min() < 0.0 ? make_xbest(c) : make_full_vector<scalar_t>(dims, 0.0);
        check_solution(program, expected_t{xbest});
    }
}

UTEST_CASE(program9)
{
    const auto make_sorted = [](const vector_t& c)
    {
        std::vector<std::pair<scalar_t, tensor_size_t>> values;
        values.reserve(static_cast<size_t>(c.size()));
        for (tensor_size_t i = 0; i < c.size(); ++i)
        {
            values.emplace_back(c(i), i);
        }
        std::sort(values.begin(), values.end());
        return values;
    };

    // exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over a unit box with a total budget constraint:
    //  min c.dot(x) s.t. 1.dot(x) = alpha, 0 <= x <= 1,
    //  where alpha is an integer between 0 and n.
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (tensor_size_t alpha = 0; alpha <= dims; ++alpha)
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",alpha=", alpha, ",x.sum()==alpha"));

            const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
            const auto a = make_full_vector<scalar_t>(dims, 1.0);
            const auto v = make_sorted(c);

            const auto program = make_linear(c, make_equality(a, static_cast<scalar_t>(alpha)), make_greater(dims, 0.0),
                                             make_less(dims, 1.0));

            auto xbest = make_full_vector<scalar_t>(dims, 0.0);
            for (tensor_size_t i = 0; i < alpha; ++i)
            {
                const auto [value, index] = v[static_cast<size_t>(i)];
                xbest(index)              = 1.0;
            }
            check_solution(program, expected_t{xbest});
        }
    }

    // exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over a unit box with a total budget constraint:
    //  min c.dot(x) s.t. 1.dot(x) <= alpha, 0 <= x <= 1,
    //  where alpha is an integer between 0 and n.
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (tensor_size_t alpha = 0; alpha <= dims; ++alpha)
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",alpha=", alpha, ",x.sum()<=alpha"));

            const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
            const auto a = make_full_vector<scalar_t>(dims, 1.0);
            const auto v = make_sorted(c);

            const auto program = make_linear(c, make_inequality(a, static_cast<scalar_t>(alpha)),
                                             make_greater(dims, 0.0), make_less(dims, 1.0));
            const auto estatus = alpha == 0 ? solver_status::unfeasible : solver_status::converged;

            auto xbest = make_full_vector<scalar_t>(dims, 0.0);
            for (tensor_size_t i = 0, count = 0; i < dims && count < alpha; ++i)
            {
                const auto [value, index] = v[static_cast<size_t>(i)];
                if (value <= 0.0)
                {
                    ++count;
                    xbest(index) = 1.0;
                }
            }
            check_solution(program, expected_t{xbest}.status(estatus));
        }
    }
}

UTEST_CASE(program10)
{
    const auto make_sorted = [](const vector_t& c, const vector_t& d)
    {
        std::vector<std::pair<scalar_t, tensor_size_t>> values;
        values.reserve(static_cast<size_t>(c.size()));
        for (tensor_size_t i = 0; i < c.size(); ++i)
        {
            values.emplace_back(c(i) / d(i), i);
        }
        std::sort(values.begin(), values.end());
        return values;
    };

    // exercise 4.8 (f), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over a unit box with a weighted budget constraint:
    //  min c.dot(x) s.t. d.dot(x) = alpha, 0 <= x <= 1,
    //  where d > 0 and 0 <= alpha <= 1.dot(d).
    for (const tensor_size_t dims : {2, 3, 5})
    {
        const auto d = make_random_vector<scalar_t>(dims, 1.0, +2.0);

        for (const auto alpha : {0.0, 0.3 * d.sum(), 0.7 * d.sum(), d.sum()})
        {
            UTEST_NAMED_CASE(scat("dims=", dims, ",alpha=", alpha / d.sum()));

            const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
            const auto v = make_sorted(c, d);

            const auto program = make_linear(c, make_equality(d, alpha), make_greater(dims, 0.0), make_less(dims, 1.0));

            auto accum = 0.0;
            auto xbest = make_full_vector<scalar_t>(dims, 0.0);
            for (tensor_size_t i = 0; i < dims && accum < alpha; ++i)
            {
                [[maybe_unused]] const auto [value, index] = v[static_cast<size_t>(i)];
                if (accum + d(index) <= alpha)
                {
                    xbest(index) = 1.0;
                }
                else
                {
                    xbest(index) = (alpha - accum) / d(index);
                }
                accum += d(index);
            }
            check_solution(program, expected_t{xbest});
        }
    }
}

UTEST_CASE(program11)
{
    // exercise 4.8 (f), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // square linear program:
    //  min c.dot(x) s.t. Ax <= b,
    //  where A is square and nonsingular and A^T * c <= 0 (to be feasible).
    for (const tensor_size_t dims : {2, 3, 5})
    {
        UTEST_NAMED_CASE(scat("dims=", dims));

        const auto c = make_random_vector<scalar_t>(dims, -1.0, -0.0);
        const auto A = matrix_t::identity(dims, dims);
        const auto b = make_random_vector<scalar_t>(dims, -1.0, +1.0);

        const auto  program = make_linear(c, make_inequality(A, b));
        const auto& xbest   = b;
        check_solution(program, expected_t{xbest});
    }
}

UTEST_CASE(equality_unique_solution)
{
    // min c.dot(x) s.t. Ax = b and x >= 0,
    // where the linear equality has exactly one solution.
    for (const tensor_size_t dims : {2, 3, 5})
    {
        const auto D = make_random_matrix<scalar_t>(dims, dims);
        const auto A = D.transpose() * D + matrix_t::identity(dims, dims);
        const auto c = make_random_vector<scalar_t>(dims);
        {
            UTEST_NAMED_CASE(scat("feasible(dims=", dims, ")"));

            // the solution is feasible
            const auto x = make_random_vector<scalar_t>(dims, +1.0, +2.0);
            const auto b = A * x;

            const auto  program = make_linear(c, make_equality(A, b), make_greater(dims, 0.0));
            const auto& xbest   = x;
            check_solution(program, expected_t{xbest});
        }
        {
            UTEST_NAMED_CASE(scat("not feasible(dims=", dims, ")"));

            // the solution is not feasible
            const auto x = make_random_vector<scalar_t>(dims, -2.0, -1.0);
            const auto b = A * x;

            const auto program = make_linear(c, make_equality(A, b), make_greater(dims, 0.0));
            check_solution(program, expected_t{}.status(solver_status::stopped));
            // FIXME: check_solution(program, expected_t{}.status(solver_status::unbounded));
        }
    }
}

UTEST_END_MODULE()
