#include <iomanip>
#include <iostream>
#include <nano/solver/linprog.h>

using namespace nano;

int main(const int, char*[])
{
    // construct a linear programming objective in standard form:
    //  min c.dot(x) s.t Ax = b and x >= 0.
    //
    // in particular:
    //  min x1 + x2 + x3
    //  s.t. 2 * x1 + x2 = 4, x1 + x3 = 1, x1 >= 0, x2 >= 0, x3 >= 0.
    //
    // with solution: (1, 2, 0, 0)
    const auto n_equals = 2;
    const auto c        = make_vector<scalar_t>(1, 1, 1);
    const auto A        = make_matrix<scalar_t>(n_equals, 2, 1, 0, 1, 0, 1);
    const auto b        = make_vector<scalar_t>(4, 1);
    const auto xbest    = make_vector<scalar_t>(1, 2, 0);

    // log the optimization steps
    const auto logger = [](const linprog::problem_t& problem, const linprog::solution_t& solution)
    {
        std::cout << std::fixed << std::setprecision(12) << "i=" << solution.m_iters << ",miu=" << solution.m_miu
                  << ",KKT=" << solution.m_kkt << ",c.dot(x)=" << problem.m_c.dot(solution.m_x)
                  << ",|Ax-b|=" << (problem.m_A * solution.m_x - problem.m_b).lpNorm<Eigen::Infinity>() << std::endl;
    };

    // solve the linear programming problem
    auto solver                           = linprog::solver_t{logger};
    solver.parameter("solver::epsilon")   = 1e-15;
    solver.parameter("solver::max_iters") = 100;

    const auto problem  = linprog::problem_t{c, A, b};
    const auto solution = solver.solve(problem);

    std::cout << std::fixed << std::setprecision(12) << "solution: x=" << solution.m_x.transpose() << std::endl;

    const auto error = (solution.m_x - xbest).lpNorm<Eigen::Infinity>();
    return error < 1e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}
