#include <iomanip>
#include <iostream>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/quadratic.h>
#include <nano/solver.h>

using namespace nano;

int main(const int, char*[])
{
    // construct a quadratic program in general form:
    //  min f(x) = 1/2 * x.dot(Q * x) + c.dot(x)
    //  s.t A * x = b
    //  and G * x <= h.
    //
    // in particular:
    //  min x1^2 + 4 * x2^2 - 8 * x1 - 16 * x2
    //  s.t x1 + 2 * x2 = 12, x1 + x2 <= 10, 1 <= x1 <= 3, 1 <= x2 <= 6.
    //
    // with solution: (3, 4.5)
    const auto n_equals   = 1;
    const auto n_inequals = 1;
    const auto Q          = make_matrix<scalar_t>(2, 2, 0, 0, 8);
    const auto c          = make_vector<scalar_t>(-8, -16);
    const auto A          = make_matrix<scalar_t>(n_equals, 1, 2);
    const auto b          = make_vector<scalar_t>(12);
    const auto G          = make_matrix<scalar_t>(n_inequals, 1, 1);
    const auto h          = make_vector<scalar_t>(10);
    const auto l          = make_vector<scalar_t>(1, 1);
    const auto u          = make_vector<scalar_t>(3, 6);
    const auto xbest      = make_vector<scalar_t>(3, 4.5);

    // solve the quadratic program
    auto solver = solver_t::all().get("ipm");
    assert(solver != nullptr);
    solver->parameter("solver::epsilon")   = 1e-12;
    solver->parameter("solver::max_evals") = 100;

    auto program = quadratic_program_t{"qp", Q, c};
    critical(A * program.variable() == b);
    critical(G * program.variable() <= h);
    critical(l <= program.variable());
    critical(program.variable() <= u);

    const auto logger = make_stdout_logger();
    const auto state  = solver->minimize(program, make_random_vector<scalar_t>(program.size()), logger);

    std::cout << std::fixed << std::setprecision(12) << "solution: x=" << state.x().transpose() << std::endl;

    assert(state.status() == solver_status::kkt_optimality_test);
    assert(close(state.x(), xbest, 1e-10));

    const auto error = (state.x() - xbest).lpNorm<Eigen::Infinity>();
    return error < 1e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}
