#include <nano/solver/penalty.h>

using namespace nano;

template <typename tpenalty>
solver_penalty_t<tpenalty>::solver_penalty_t(rsolver_t&& solver)
    : m_solver(solver)
{
    // FIXME: configurable solver
    register_parameter(parameter_t::make_scalar("solver::penalty::c0", 0.0, LT, 1.0, LE, 1e+3));
    register_parameter(parameter_t::make_scalar("solver::penalty::gamma", 1.0, LT, 2.0, LE, 1e+3));
}

template <typename tpenalty>
solver_state_t solver_penalty_t<tpenalty>::minimize(const function_t& function, const vector_t& x0) const
{
    const auto c0 = parameter("solver::penalty

    auto solver = solver_t::all().get("lbfgs");
    solver->parameter("solver::max
}

template class solver_penalty_t<linear_penalty_function_t>;
template class solver_penalty_t<quadratic_penalty_function_t>;
