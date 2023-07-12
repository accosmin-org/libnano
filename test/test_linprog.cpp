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

UTEST_CASE(program1)
{
    // see example 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto b = make_vector<scalar_t>(5, 8);

    const auto prog = linprog::problem_t{c, A, b};
    UTEST_CHECK(prog.feasible(make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0), 1e-12));
    UTEST_CHECK(prog.feasible(make_vector<scalar_t>(0.0, 4.0, 1.0, 6.0), 1e-12));
    UTEST_CHECK(prog.feasible(make_vector<scalar_t>(2.0, 2.0, 1.0, 3.0), 1e-12));

    const auto solution0 = linprog::make_starting_point(prog);
    UTEST_CHECK_GREATER(solution0.m_x.minCoeff(), 0.0);
    UTEST_CHECK_GREATER(solution0.m_s.minCoeff(), 0.0);

    const auto fbest    = -52 / 3.0;
    const auto xbest    = make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0);
    const auto solution = linprog::solve(prog, make_logger());
    UTEST_CHECK(solution.converged());
    UTEST_CHECK_LESS(solution.m_iters, 10);
    UTEST_CHECK_CLOSE(solution.m_x, xbest, 1e-12);
    UTEST_CHECK_CLOSE(c.dot(solution.m_x), fbest, 1e-12);
}

UTEST_CASE(program2)
{
    // see exercise 14.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(1, 0);
    const auto A = make_matrix<scalar_t>(1, 1, 1);
    const auto b = make_vector<scalar_t>(1);

    const auto prog = linprog::problem_t{c, A, b};
    UTEST_CHECK(prog.feasible(make_vector<scalar_t>(0.0, 1.0), 1e-12));
    UTEST_CHECK(prog.feasible(make_vector<scalar_t>(1.0, 0.0), 1e-12));
    UTEST_CHECK(prog.feasible(make_vector<scalar_t>(0.1, 0.9), 1e-12));

    const auto solution0 = linprog::make_starting_point(prog);
    UTEST_CHECK_GREATER(solution0.m_x.minCoeff(), 0.0);
    UTEST_CHECK_GREATER(solution0.m_s.minCoeff(), 0.0);

    const auto fbest    = 0.0;
    const auto xbest    = make_vector<scalar_t>(0.0, 1.0);
    const auto solution = linprog::solve(prog, make_logger());
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

    const auto prog     = linprog::problem_t{c, A, b};
    const auto solution = linprog::solve(prog, make_logger());
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

    const auto prog     = linprog::problem_t{c, A, b};
    const auto solution = linprog::solve(prog, make_logger());
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

    const auto prog     = linprog::problem_t{c, A, b};
    const auto solution = linprog::solve(prog, make_logger());
    UTEST_CHECK(!solution.converged());
    UTEST_CHECK(solution.diverged());
    UTEST_CHECK_LESS(solution.m_iters, 10);
}

UTEST_END_MODULE()
