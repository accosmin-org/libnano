#include <nano/solver/linprog.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
auto make_logger()
{
    const auto op = [](const linprog::problem_t& problem, const linprog::solution_t& solution)
    {
        std::cout << std::fixed << std::setprecision(16) << "i=" << solution.m_iters << ",miu=" << solution.m_miu
                  << ",KKT=" << solution.m_kkt << ",c.dot(x)=" << problem.m_c.dot(solution.m_x)
                  << ",|Ax-b|=" << (problem.m_A * solution.m_x - problem.m_b).lpNorm<Eigen::Infinity>() << std::endl;
    };
    return linprog::logger_t{op};
}

template <typename tproblem>
auto check_solution(const tproblem& problem, const linprog::solution_t& solution, const vector_t& xbest,
                    const scalar_t epsilon = 1e-10)
{
    UTEST_CHECK(solution.converged(epsilon));
    UTEST_CHECK_LESS(solution.m_miu, epsilon);
    UTEST_CHECK_LESS(solution.m_kkt, epsilon);
    UTEST_CHECK_LESS(solution.m_iters, 20);
    UTEST_CHECK_CLOSE(solution.m_x, xbest, epsilon);
    UTEST_CHECK(problem.feasible(xbest, epsilon));
}
} // namespace

UTEST_BEGIN_MODULE(test_linprog)

UTEST_CASE(solution)
{
    auto solution = linprog::solution_t{};
    UTEST_CHECK(!solution.converged());

    solution.m_kkt = std::numeric_limits<scalar_t>::quiet_NaN();
    UTEST_CHECK(!solution.converged());

    solution.m_kkt = 1e-40;
    UTEST_CHECK(solution.converged());

    solution.m_kkt = 0.0;
    UTEST_CHECK(solution.converged());
}

UTEST_CASE(standard_problem)
{
    const auto c = make_vector<scalar_t>(1, 1);
    const auto A = make_matrix<scalar_t>(1, 2, 1);
    const auto b = make_vector<scalar_t>(5);

    const auto epsilon = 1e-12;
    const auto problem = linprog::problem_t{c, A, b};
    UTEST_CHECK(!problem.feasible(make_vector<scalar_t>(-1, 1), epsilon));
    UTEST_CHECK(!problem.feasible(make_vector<scalar_t>(1, 1), epsilon));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(2, 1), epsilon));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(1, 3), epsilon));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(2.5, 0), epsilon));
}

UTEST_CASE(general_problem)
{
    const auto c = make_vector<scalar_t>(1, 1);
    const auto A = make_matrix<scalar_t>(1, 2, 1);
    const auto b = make_vector<scalar_t>(5);
    const auto G = make_matrix<scalar_t>(2, 1, 0, 1, 2);
    const auto h = make_vector<scalar_t>(3, 5);

    const auto epsilon = 1e-12;
    const auto problem = linprog::general_problem_t{c, A, b, G, h};
    UTEST_CHECK(!problem.feasible(make_vector<scalar_t>(-1, 1), epsilon));
    UTEST_CHECK(!problem.feasible(make_vector<scalar_t>(1, 1), epsilon));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(2, 1), epsilon));
    UTEST_CHECK(!problem.feasible(make_vector<scalar_t>(1, 3), epsilon));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(2.5, 0), epsilon));

    const auto standard   = problem.transform();
    const auto expected_c = make_vector<scalar_t>(1, 1, -1, -1, 0, 0);
    const auto expected_A = make_matrix<scalar_t>(3, 2, 1, -2, -1, 0, 0, 1, 0, -1, 0, 1, 0, 1, 2, -1, -2, 0, 1);
    const auto expected_b = make_vector<scalar_t>(5, 3, 5);
    UTEST_CHECK_CLOSE(standard.m_c, expected_c, epsilon);
    UTEST_CHECK_CLOSE(standard.m_A, expected_A, epsilon);
    UTEST_CHECK_CLOSE(standard.m_b, expected_b, epsilon);
}

UTEST_CASE(inequality_problem)
{
    const auto c = make_vector<scalar_t>(1, 1);
    const auto A = make_matrix<scalar_t>(1, 2, 1);
    const auto b = make_vector<scalar_t>(5);

    const auto epsilon = 1e-12;
    const auto problem = linprog::inequality_problem_t{c, A, b};
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(-1, 1), epsilon));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(1, 1), epsilon));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(2, 1), epsilon));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(1, 3), epsilon));
    UTEST_CHECK(!problem.feasible(make_vector<scalar_t>(2, 2), epsilon));

    const auto standard   = problem.transform();
    const auto expected_c = make_vector<scalar_t>(1, 1, -1, -1, 0);
    const auto expected_A = make_matrix<scalar_t>(1, 2, 1, -2, -1, 1);
    const auto expected_b = make_vector<scalar_t>(5);
    UTEST_CHECK_CLOSE(standard.m_c, expected_c, epsilon);
    UTEST_CHECK_CLOSE(standard.m_A, expected_A, epsilon);
    UTEST_CHECK_CLOSE(standard.m_b, expected_b, epsilon);
}

UTEST_CASE(program1)
{
    // see example 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto b = make_vector<scalar_t>(5, 8);

    const auto problem  = linprog::problem_t{c, A, b};
    const auto solution = linprog::solve(problem, make_logger());
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0), 1e-12));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(0.0, 4.0, 1.0, 6.0), 1e-12));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(2.0, 2.0, 1.0, 3.0), 1e-12));

    const auto xbest = make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0);
    check_solution(problem, solution, xbest);
}

UTEST_CASE(program2)
{
    // see exercise 14.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(1, 0);
    const auto A = make_matrix<scalar_t>(1, 1, 1);
    const auto b = make_vector<scalar_t>(1);

    const auto problem  = linprog::problem_t{c, A, b};
    const auto solution = linprog::solve(problem, make_logger());
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(0.0, 1.0), 1e-12));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(1.0, 0.0), 1e-12));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(0.1, 0.9), 1e-12));

    const auto xbest = make_vector<scalar_t>(0.0, 1.0);
    check_solution(problem, solution, xbest);
}

UTEST_CASE(program3)
{
    // NB: unbounded problem!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(1, 0, 1, 1);
    const auto b = make_vector<scalar_t>(2);

    const auto problem  = linprog::problem_t{c, A, b};
    const auto solution = linprog::solve(problem, make_logger());
    UTEST_CHECK(!solution.converged());
    UTEST_CHECK_LESS(solution.m_iters, 10);
}

UTEST_CASE(program4)
{
    // NB: unfeasible problem!
    const auto c = make_vector<scalar_t>(-1, 0);
    const auto A = make_matrix<scalar_t>(2, 0, 1, 1, 0);
    const auto b = make_vector<scalar_t>(-1, -1);

    const auto problem  = linprog::problem_t{c, A, b};
    const auto solution = linprog::solve(problem, make_logger());
    UTEST_CHECK(!solution.converged());
    UTEST_CHECK_LESS(solution.m_iters, 10);
}

UTEST_CASE(program5)
{
    // NB: unfeasible problem!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(3, 0, 1, 1, 0, 0, 1, 0, 1, 0);
    const auto b = make_vector<scalar_t>(1, 1, 1);

    const auto problem  = linprog::problem_t{c, A, b};
    const auto solution = linprog::solve(problem, make_logger());
    UTEST_CHECK(!solution.converged());
    UTEST_CHECK_LESS(solution.m_iters, 10);
}

UTEST_CASE(program6)
{
    // exercise 4.8 (b), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over a halfspace:
    //  min c.dot(x) s.t. a.dot(x) <= b,
    //  where c = lambda * a.
    for (const tensor_size_t dims : {1, 7, 17})
    {
        for (const auto lambda : {-1.0, -1.42, -4.2, -42.1})
        {
            const auto a = make_random_vector<scalar_t>(dims, +1.0, +2.0);
            const auto b = urand<scalar_t>(-1.0, +1.0, make_rng());
            const auto c = lambda * a;

            const auto problem  = linprog::inequality_problem_t{c, map_matrix(a.data(), 1, dims), map_vector(&b, 1)};
            const auto solution = linprog::solve(problem, make_logger());

            const auto fbest = lambda * b;
            UTEST_CHECK(solution.converged());
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
    for (const tensor_size_t dims : {1, 7, 17})
    {
        const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
        const auto l = make_random_vector<scalar_t>(dims, -1.0, +1.0);
        const auto u = make_random_vector<scalar_t>(dims, +1.0, +3.0);

        auto A = matrix_t{2 * dims, dims};
        A.block(0, 0, dims, dims).setIdentity();
        A.block(dims, 0, dims, dims) = -matrix_t::Identity(dims, dims);

        auto b                = vector_t{2 * dims};
        b.segment(0, dims)    = u;
        b.segment(dims, dims) = -l;

        const auto problem  = linprog::inequality_problem_t{c, std::move(A), std::move(b)};
        const auto solution = linprog::solve(problem, make_logger());

        const auto xbest = vector_t{l.array() * c.array().max(0.0).sign() - u.array() * c.array().min(0.0).sign()};
        check_solution(problem, solution, xbest);
        UTEST_CHECK_GREATER_EQUAL((solution.m_x - l).minCoeff(), -1e-10);
        UTEST_CHECK_GREATER_EQUAL((u - solution.m_x).minCoeff(), -1e-10);
    }
}

UTEST_CASE(program8)
{
    const auto make_xbest = [&](const vector_t& c)
    {
        const auto dims = c.size();
        const auto cmin = c.minCoeff();

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
    for (const tensor_size_t dims : {2, 7, 17})
    {
        const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
        const auto A = make_full_matrix<scalar_t>(1, dims, 1.0);
        const auto b = make_vector<scalar_t>(1.0);

        const auto problem  = linprog::problem_t{c, A, b};
        const auto solution = linprog::solve(problem, make_logger());

        const auto xbest = make_xbest(c);
        check_solution(problem, solution, xbest);
    }

    // exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over the probability simplex:
    //  min c.dot(x) s.t. 1.dot(x) <= 1, x >= 0.
    for (const tensor_size_t dims : {2, 7, 17})
    {
        const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);

        auto A = matrix_t(dims + 1, dims);
        A.row(0).setConstant(1.0);
        A.block(1, 0, dims, dims).setZero();
        A.block(1, 0, dims, dims).diagonal().setConstant(-1.0);

        auto b = vector_t(dims + 1);
        b(0)   = 1.0;
        b.segment(1, dims).setConstant(0.0);

        const auto problem  = linprog::inequality_problem_t{c, std::move(A), std::move(b)};
        const auto solution = linprog::solve(problem, make_logger());

        const auto xbest = c.minCoeff() < 0.0 ? make_xbest(c) : make_full_vector<scalar_t>(dims, 0.0);
        check_solution(problem, solution, xbest);
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
    for (const tensor_size_t dims : {2, 3, 7})
    {
        for (tensor_size_t alpha = 0; alpha <= dims; ++alpha)
        {
            const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
            const auto b = make_vector<scalar_t>(alpha);
            const auto A = make_full_matrix<scalar_t>(1, dims, 1.0);
            const auto v = make_sorted(c);

            auto G = matrix_t{2 * dims, dims};
            G.block(0, 0, dims, dims).setIdentity();
            G.block(dims, 0, dims, dims) = -matrix_t::Identity(dims, dims);

            auto h                        = vector_t{2 * dims};
            h.segment(0, dims).array()    = 1;
            h.segment(dims, dims).array() = 0;

            const auto problem  = linprog::general_problem_t{c, A, b, std::move(G), std::move(h)};
            const auto solution = linprog::solve(problem, make_logger());

            auto xbest = make_full_vector<scalar_t>(dims, 0.0);
            for (tensor_size_t i = 0; i < alpha; ++i)
            {
                const auto [value, index] = v[static_cast<size_t>(i)];
                xbest(index)              = 1.0;
            }
            check_solution(problem, solution, xbest);
        }
    }

    // exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // minimizing a linear function over a unit box with a total budget constraint:
    //  min c.dot(x) s.t. 1.dot(x) <= alpha, 0 <= x <= 1,
    //  where alpha is an integer between 0 and n.
    for (const tensor_size_t dims : {2, 3, 7})
    {
        for (tensor_size_t alpha = 0; alpha <= dims; ++alpha)
        {
            const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
            const auto v = make_sorted(c);

            auto A           = matrix_t{1 + 2 * dims, dims};
            A.row(0).array() = 1.0;
            A.block(1, 0, dims, dims).setIdentity();
            A.block(1 + dims, 0, dims, dims) = -matrix_t::Identity(dims, dims);

            auto b                            = vector_t{1 + 2 * dims};
            b(0)                              = static_cast<scalar_t>(alpha);
            b.segment(1, dims).array()        = 1;
            b.segment(1 + dims, dims).array() = 0;

            const auto problem  = linprog::inequality_problem_t{c, A, b};
            const auto solution = linprog::solve(problem, make_logger());

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
            check_solution(problem, solution, xbest);
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
    for (const tensor_size_t dims : {2, 3, 7})
    {
        const auto d = make_random_vector<scalar_t>(dims, 1.0, +2.0);

        for (const auto alpha : {0.0, 0.3 * d.sum(), 0.7 * d.sum(), d.sum()})
        {
            const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
            const auto b = make_vector<scalar_t>(alpha);
            const auto v = make_sorted(c, d);

            auto A   = matrix_t{1, dims};
            A.row(0) = d;

            auto G = matrix_t{2 * dims, dims};
            G.block(0, 0, dims, dims).setIdentity();
            G.block(dims, 0, dims, dims) = -matrix_t::Identity(dims, dims);

            auto h                        = vector_t{2 * dims};
            h.segment(0, dims).array()    = 1;
            h.segment(dims, dims).array() = 0;

            const auto problem  = linprog::general_problem_t{c, A, b, std::move(G), std::move(h)};
            const auto solution = linprog::solve(problem, make_logger());

            auto accum = 0.0;
            auto xbest = make_full_vector<scalar_t>(dims, 0.0);
            for (tensor_size_t i = 0; i < dims && accum < alpha; ++i)
            {
                [[maybe_unused]] const auto [_, index] = v[static_cast<size_t>(i)];
                if (accum + d(index) > alpha)
                {
                    xbest(index) = (alpha - accum) / d(index);
                }
                else
                {
                    xbest(index) = 1.0;
                }
                accum += d(index);
            }
            check_solution(problem, solution, xbest, 1e-8);
        }
    }
}

UTEST_CASE(program11)
{
    // exercise 4.8 (f), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    // square linear problem:
    //  min c.dot(x) s.t. Ax <= b,
    //  where A is square and nonsingular and A^-T * c <= 0 (to be feasible).
    for (const tensor_size_t dims : {2, 3, 7})
    {
        const auto c = make_random_vector<scalar_t>(dims, -1.0, -0.0);
        const auto A = matrix_t::Identity(dims, dims);
        const auto b = make_random_vector<scalar_t>(dims, -1.0, +1.0);

        const auto problem  = linprog::inequality_problem_t{c, A, b};
        const auto solution = linprog::solve(problem, make_logger());

        const auto& xbest = b;
        check_solution(problem, solution, xbest);
    }
}

// TODO: process problems when A is not row full rank! - can add unit tests explicitly for this

UTEST_END_MODULE()
