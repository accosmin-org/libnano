#include <iomanip>
#include <utest/utest.h>
#include <nano/numeric.h>
#include <nano/solver/stochastic.h>
#include <nano/function/geometric.h>

using namespace nano;

template <typename tsolver>
static auto make_solver()
{
    auto solver = tsolver{};
    solver.batch0(1);
    solver.batchr(1.0);
    solver.epsilon(1e-3);
    solver.max_iterations(100);
    return solver;
}

static void check_solver(const function_t& function, const char* solver_name, solver_t& solver)
{
    std::stringstream stream;
    tensor_size_t iterations = 0;
    solver.logger([&] (const solver_state_t& state)
    {
        ++ iterations;
        stream << std::fixed << std::setprecision(6)
            << "\tdescent: i=" << state.m_iterations << ",f=" << state.f << ",g=" << state.convergence_criterion()
            << "[" << state.m_status << "],calls=" << state.m_fcalls << "/" << state.m_gcalls
            << ",lrate=" << state.lrate << ",decay=" << state.decay << ".\n";
        return true;
    });

    const vector_t x0 = vector_t::Ones(function.size());
    const auto state = solver.minimize(function, x0);
    std::cout << solver_name << ": iters=" << state.m_iterations << std::fixed << std::setprecision(6)
        << ",f=" << state.f << ",g=" << state.convergence_criterion()
        << "[" << state.m_status << "],calls=" << state.m_fcalls << "/" << state.m_gcalls
        << ",lrate=" << state.lrate << ",decay=" << state.decay << ".\n";

    UTEST_CHECK(state);
    UTEST_CHECK_LESS(state.convergence_criterion(), solver.epsilon());
    UTEST_CHECK_EQUAL(state.m_status, solver_state_t::status::converged);
    UTEST_CHECK_EQUAL(iterations, state.m_iterations);

    if (state.m_status != solver_state_t::status::converged)
    {
        std::cout << stream.str();
    }
}

UTEST_BEGIN_MODULE(test_solver_stoch)

UTEST_CASE(sgd)
{
    auto solver = make_solver<solver_sgd_t>();
    const auto function = function_geometric_optimization_t{4, 1024};

    check_solver(function, "sgd", solver);
}

UTEST_CASE(asgd)
{
    auto solver = make_solver<solver_asgd_t>();
    const auto function = function_geometric_optimization_t{4, 1024};

    check_solver(function, "asgd", solver);
}

UTEST_END_MODULE()
