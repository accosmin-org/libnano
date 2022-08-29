#include <nano/core/numeric.h>
#include <nano/solver.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_solver(const char* name = "cgd", scalar_t epsilon = 1e-8, int max_evals = 10000)
{
    auto solver = solver_t::all().get(name);
    UTEST_REQUIRE(solver);
    solver->parameter("solver::epsilon")   = epsilon;
    solver->parameter("solver::max_evals") = max_evals;
    return solver;
}

static void setup_logger(solver_t& solver, std::stringstream& stream, tensor_size_t& iterations)
{
    solver.logger(
        [&](const solver_state_t& state)
        {
            ++iterations;
            stream << "\tdescent: " << state << ".\n";
            return true;
        });

    solver.lsearch0_logger(
        [&](const solver_state_t& state0, const scalar_t t)
        {
            stream << "\t\tlsearch(0): t=" << state0.t << ",f=" << state0.f << ",g=" << state0.convergence_criterion()
                   << ",t=" << t << ".\n";
        });

    const auto [c1, c2] = solver.parameter("solver::tolerance").value_pair<scalar_t>();

    solver.lsearchk_logger(
        [&, c1 = c1, c2 = c2](const solver_state_t& state0, const solver_state_t& state)
        {
            stream << "\t\tlsearch(t):t=" << state.t << ",f=" << state.f << ",g=" << state.convergence_criterion()
                   << ",armijo=" << state.has_armijo(state0, c1) << ",wolfe=" << state.has_wolfe(state0, c2)
                   << ",swolfe=" << state.has_strong_wolfe(state0, c2) << ".\n";
        });
}

[[maybe_unused]] static auto check_minimize(solver_t& solver, const string_t& solver_id, const function_t& function,
                                            const vector_t& x0, tensor_size_t max_evals = 50000,
                                            scalar_t epsilon = 1e-6, bool converges = true)
{
    const auto old_n_failures = utest_n_failures.load();
    const auto state0         = solver_state_t{function, x0};

    const auto lsearch0_id = solver.monotonic() ? solver.lsearch0_id() : "N/A";
    const auto lsearchk_id = solver.monotonic() ? solver.lsearchk_id() : "N/A";

    std::stringstream stream;
    stream << std::fixed << std::setprecision(16) << function.name() << " " << solver_id << "[" << lsearch0_id << ","
           << lsearchk_id << "]\n"
           << ":x0=[" << state0.x.transpose() << "],f0=" << state0.f << ",g0=" << state0.convergence_criterion()
           << "\n";

    tensor_size_t iterations = 0;
    setup_logger(solver, stream, iterations);

    // minimize
    solver.parameter("solver::epsilon")   = epsilon;
    solver.parameter("solver::max_evals") = max_evals;
    auto state                            = solver.minimize(function, x0);
    UTEST_CHECK(state);

    // check function value decrease
    UTEST_CHECK_LESS_EQUAL(state.f, state0.f + epsilon1<scalar_t>());

    // check convergence
    if (function.smooth() && solver.monotonic())
    {
        UTEST_CHECK_LESS(state.convergence_criterion(), epsilon);
    }
    UTEST_CHECK_EQUAL(state.status, converges ? solver_status::converged : solver_status::max_iters);
    UTEST_CHECK_EQUAL(iterations, state.inner_iters);

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }

    return state;
}
