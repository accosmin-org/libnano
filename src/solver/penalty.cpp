#include <nano/critical.h>
#include <nano/function/penalty.h>
#include <solver/penalty.h>

using namespace nano;

solver_penalty_t::solver_penalty_t(string_t id)
    : solver_t(std::move(id))
{
    register_parameter(parameter_t::make_scalar("solver::penalty::epsilon0", 0.0, LT, 1e-8, LE, 1e-2));
    register_parameter(parameter_t::make_scalar("solver::penalty::epsilonK", 0.0, LT, 0.90, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::penalty::eta", 1.0, LT, 5.0, LE, 1e+3));
    register_parameter(parameter_t::make_scalar("solver::penalty::penalty0", 0.0, LT, 10.0, LE, 1e+3));

    // NB: stops if no significant improvement in two outer iterations!
    parameter("solver::patience") = 2;

    // NB: more iterations are needed by default!
    parameter("solver::max_evals") = 50 * parameter("solver::max_evals").value<tensor_size_t>();
}

solver_state_t solver_penalty_t::minimize(penalty_function_t& penalty_function, const vector_t& x0,
                                          const logger_t& logger) const
{
    const auto max_evals      = parameter("solver::max_evals").value<tensor_size_t>();
    const auto base_solver_id = parameter("solver::penalty::base_solver_id").value<string_t>();
    const auto eta            = parameter("solver::penalty::eta").value<scalar_t>();
    const auto epsilon0       = parameter("solver::penalty::epsilon0").value<scalar_t>();
    const auto epsilonK       = parameter("solver::penalty::epsilonK").value<scalar_t>();
    const auto penalty0       = parameter("solver::penalty::penalty0").value<scalar_t>();

    auto penalty = penalty0;
    auto bstate  = solver_state_t{penalty_function.function(), x0};
    auto solver  = solver_t::all().get(base_solver_id);
    auto outer   = 0;

    critical(solver != nullptr, scat("[solver-", type_id(), "]: invalid solver id <", base_solver_id, ">!"));
    solver->parameter("solver::epsilon") = epsilon0;

    while (penalty_function.fcalls() + penalty_function.gcalls() < max_evals && (outer++) < 1000)
    {
        // solve the penalty problem
        penalty_function.penalty(penalty);
        const auto cstate = solver->minimize(penalty_function, bstate.x(), logger);

        // increase penalty until the solution is bounded
        const auto iter_ok = cstate.valid();
        if (!iter_ok)
        {
            penalty *= eta;
            continue;
        }

        // check convergence
        bstate.update(cstate.x());
        if (done_value_test(bstate, iter_ok, logger))
        {
            break;
        }

        // update penalty parameter
        penalty *= eta;
        solver->more_precise(epsilonK);
    }

    return bstate;
}

solver_linear_penalty_t::solver_linear_penalty_t()
    : solver_penalty_t("linear-penalty")
{
    register_parameter(parameter_t::make_string("solver::penalty::base_solver_id", "osga"));
}

rsolver_t solver_linear_penalty_t::clone() const
{
    return std::make_unique<solver_linear_penalty_t>(*this);
}

solver_state_t solver_linear_penalty_t::do_minimize(const function_t& function, const vector_t& x0,
                                                    const logger_t& logger) const
{
    auto penalty_function = linear_penalty_function_t{function};

    return minimize(penalty_function, x0, logger);
}

solver_quadratic_penalty_t::solver_quadratic_penalty_t()
    : solver_penalty_t("quadratic-penalty")
{
    register_parameter(parameter_t::make_string("solver::penalty::base_solver_id", "lbfgs"));
}

rsolver_t solver_quadratic_penalty_t::clone() const
{
    return std::make_unique<solver_quadratic_penalty_t>(*this);
}

solver_state_t solver_quadratic_penalty_t::do_minimize(const function_t& function, const vector_t& x0,
                                                       const logger_t& logger) const
{
    auto penalty_function = quadratic_penalty_function_t{function};

    return minimize(penalty_function, x0, logger);
}
