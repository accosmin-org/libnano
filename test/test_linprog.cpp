#include <nano/solver/linprog.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
auto make_logger()
{
    const auto op = [](const linprog::solution_t& solution)
    {
        std::cout << std::fixed << std::setprecision(16) << "i=" << solution.m_iters << ",miu=" << solution.m_miu
                  << ",x=" << solution.m_x.transpose() << std::endl;
    };
    return linprog::logger_t{op};
}
} // namespace

UTEST_BEGIN_MODULE(test_linprog)

UTEST_CASE(solution)
{
    auto solution = linprog::solution_t{};
    UTEST_CHECK(!solution.converged());
    UTEST_CHECK(solution.diverged());

    solution.m_miu = std::numeric_limits<scalar_t>::quiet_NaN();
    UTEST_CHECK(!solution.converged());
    UTEST_CHECK(solution.diverged());

    solution.m_miu = std::numeric_limits<scalar_t>::epsilon();
    UTEST_CHECK(solution.converged());
    UTEST_CHECK(!solution.diverged());

    solution.m_miu = 0.0;
    UTEST_CHECK(solution.converged());
    UTEST_CHECK(!solution.diverged());
}

UTEST_CASE(program1)
{
    // see example 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto b = make_vector<scalar_t>(5, 8);

    const auto problem = linprog::problem_t{c, A, b};
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0), 1e-12));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(0.0, 4.0, 1.0, 6.0), 1e-12));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(2.0, 2.0, 1.0, 3.0), 1e-12));

    const auto fbest    = -52 / 3.0;
    const auto xbest    = make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0);
    const auto solution = linprog::solve(problem, make_logger());
    UTEST_CHECK(solution.converged());
    UTEST_CHECK_LESS(solution.m_iters, 20);
    UTEST_CHECK_CLOSE(solution.m_x, xbest, 1e-12);
    UTEST_CHECK_CLOSE(c.dot(solution.m_x), fbest, 1e-12);
}

UTEST_CASE(program2)
{
    // see exercise 14.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(1, 0);
    const auto A = make_matrix<scalar_t>(1, 1, 1);
    const auto b = make_vector<scalar_t>(1);

    const auto problem = linprog::problem_t{c, A, b};
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(0.0, 1.0), 1e-12));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(1.0, 0.0), 1e-12));
    UTEST_CHECK(problem.feasible(make_vector<scalar_t>(0.1, 0.9), 1e-12));

    const auto fbest    = 0.0;
    const auto xbest    = make_vector<scalar_t>(0.0, 1.0);
    const auto solution = linprog::solve(problem, make_logger());
    UTEST_CHECK(solution.converged());
    UTEST_CHECK_LESS(solution.m_iters, 10);
    UTEST_CHECK_CLOSE(solution.m_x, xbest, 1e-12);
    UTEST_CHECK_CLOSE(c.dot(solution.m_x), fbest, 1e-12);
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
    UTEST_CHECK(solution.diverged());
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
    UTEST_CHECK(solution.diverged());
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
    UTEST_CHECK(solution.diverged());
    UTEST_CHECK_LESS(solution.m_iters, 10);
}

UTEST_CASE(program6)
{
    const auto make_problem = [](const vector_t& c, const vector_t& a, const scalar_t b)
    {
        assert(a.size() == c.size());

        const auto dims = c.size();

        auto c2                = vector_t{2 * dims + 1};
        c2.segment(0, dims)    = c;
        c2.segment(dims, dims) = -c;
        c2(2 * dims)           = 0.0;

        auto A                       = matrix_t{1, 2 * dims + 1};
        A.row(0).segment(0, dims)    = a;
        A.row(0).segment(dims, dims) = -a;
        A(0, 2 * dims)               = 1.0;

        auto b2 = vector_t{1};
        b2(0)   = b;

        return linprog::problem_t{std::move(c2), std::move(A), std::move(b2)};
    };

    // exercise 4.8 (b), see "Convex Optimization", by S. Boyd and L. Vanderberghe
    //  min c.dot(x) s.t. a.dot(x) <= b
    //  where c = lambda * a
    for (const tensor_size_t dims : {1, 7, 17, 33})
    {
        for (const auto lambda : {-1.0, -1.42, -4.2, -42.1})
        {
            const auto a = make_random_vector<scalar_t>(dims, +1.0, +2.0);
            const auto b = urand<scalar_t>(-1.0, +1.0, make_rng());
            const auto c = lambda * a;

            // TODO: utility to transform generic inequality LPs to standard form (problem & solution)
            const auto problem  = make_problem(c, a, b);
            const auto solution = linprog::solve(problem, make_logger());

            const auto xbest = vector_t{solution.m_x.segment(0, dims) - solution.m_x.segment(dims, dims)};
            const auto sbest = solution.m_x(2 * dims);
            const auto fbest = lambda * b;

            UTEST_CHECK_CLOSE(solution.m_x.dot(problem.m_c), fbest, 1e-12);
            UTEST_CHECK_CLOSE(xbest.dot(c), fbest, 1e-12);
            UTEST_CHECK_CLOSE(sbest, 0.0, 1e-12);
        }
    }
}

UTEST_END_MODULE()
