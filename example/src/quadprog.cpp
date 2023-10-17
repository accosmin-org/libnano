#include <iomanip>
#include <iostream>
#include <nano/program/solver.h>

using namespace nano;
using namespace nano::program;

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

    // log the optimization steps
    const auto logger = [](const solver_state_t& state)
    {
        std::cout << std::fixed << std::setprecision(10) << "i=" << state.m_iters << ",fx=" << state.m_fx
                  << ",eta=" << state.m_eta << ",rdual=" << state.m_rdual.lpNorm<Eigen::Infinity>()
                  << ",rcent=" << state.m_rcent.lpNorm<Eigen::Infinity>()
                  << ",rprim=" << state.m_rprim.lpNorm<Eigen::Infinity>() << ",rcond=" << state.m_ldlt_rcond
                  << (state.m_ldlt_positive ? "(+)" : "(-)") << "[" << state.m_status << "]" << std::endl;
        return true;
    };

    // solve the quadratic program
    auto solver                           = solver_t{logger};
    solver.parameter("solver::epsilon")   = 1e-12;
    solver.parameter("solver::max_iters") = 100;

    const auto program =
        make_quadratic(Q, c, make_equality(A, b), make_inequality(G, h), make_greater(l), make_less(u));
    const auto state = solver.solve(program);

    std::cout << std::fixed << std::setprecision(12) << "solution: x=" << state.m_x.transpose() << std::endl;

    assert(state.m_status == solver_status::converged);
    assert(close(state.m_x, xbest, 1e-10));

    const auto error = (state.m_x - xbest).lpNorm<Eigen::Infinity>();
    return error < 1e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}
