#include <iomanip>
#include <iostream>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/linear.h>
#include <nano/solver.h>

using namespace nano;

int main(const int, char*[])
{
    // construct a linear program in standard form:
    //  min c.dot(x)
    //  s.t Ax = b
    //  and x >= 0.
    //
    // in particular:
    //  min x1 + x2 + x3
    //  s.t 2 * x1 + x2 = 4, x1 + x3 = 1, x1 >= 0, x2 >= 0, x3 >= 0.
    //
    // with solution: (1, 2, 0, 0)
    const auto n_equals = 2;
    const auto c        = make_vector<scalar_t>(1, 1, 1);
    const auto A        = make_matrix<scalar_t>(n_equals, 2, 1, 0, 1, 0, 1);
    const auto b        = make_vector<scalar_t>(4, 1);
    const auto xbest    = make_vector<scalar_t>(1, 2, 0);

    // solve the linear program
    auto solver = solver_t::all().get("ipm");
    assert(solver != nullptr);
    solver->parameter("solver::epsilon")   = 1e-12;
    solver->parameter("solver::max_evals") = 100;

    auto program = linear_program_t{"lp", c};
    critical(A * program.variable() == b);
    critical(program.variable() >= 0.0);

    const auto logger = make_stdout_logger();
    const auto state  = solver->minimize(program, make_random_vector<scalar_t>(program.size()), logger);

    std::cout << std::fixed << std::setprecision(12) << "solution: x=" << state.x().transpose() << std::endl;

    assert(state.status() == solver_status::kkt_optimality_test);
    assert(close(state.x(), xbest, 1e-10));

    const auto error = (state.x() - xbest).lpNorm<Eigen::Infinity>();
    return error < 1e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}
