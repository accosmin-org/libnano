#include <nano/loss.h>
#include <nano/solver.h>
#include <utest/utest.h>
#include <nano/core/numeric.h>

using namespace nano;

inline auto make_loss(const char* name = "squared")
{
    auto loss = loss_t::all().get(name);
    UTEST_REQUIRE(loss);
    return loss;
}

inline auto make_solver(const char* name = "cgd", scalar_t epsilon = epsilon2<scalar_t>())
{
    auto solver = solver_t::all().get(name);
    UTEST_REQUIRE(solver);
    solver->parameter("solver::epsilon") = epsilon;
    solver->parameter("solver::max_evals") = 100;
    solver->logger([] (const solver_state_t& state)
    {
        std::cout << state << ".\n";
        return true;
    });
    return solver;
}
