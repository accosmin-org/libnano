#include <nano/solver/asgm.h>
#include <nano/core/numeric.h>

using namespace nano;

solver_asgm_t::solver_asgm_t()
{
    monotonic(false);

    register_parameter(parameter_t::make_integer("solver::asgm::patience", 2, LE, 3, LE, 100));
    register_parameter(parameter_t::make_float("solver::asgm::gamma", 1.0, LT, 5.0, LE, 100.0));
}

solver_state_t solver_asgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();
    const auto epsilon = parameter("solver::epsilon").value<scalar_t>();
    const auto gamma = parameter("solver::asgm::gamma").value<scalar_t>();
    const auto patience = parameter("solver::asgm::patience").value<int64_t>();

    auto function = make_function(function_, x0);

    auto cstate = solver_state_t{function, x0}; // current state
    auto bstate = cstate;                       // best state

    auto h = 1.0;                               // current step length ratio
    auto L = cstate.g.lpNorm<2>();              // estimation of the Lipschitz constant

    for (int64_t i = 0, last_ibest = 0; function.fcalls() < max_evals; ++ i)
    {
        cstate.d = -cstate.g / cstate.g.lpNorm<2>();
        cstate.update(cstate.x + h / L * cstate.d);

        const auto df = std::fabs(cstate.f - bstate.f);
        const auto iter_ok = static_cast<bool>(cstate);
        const auto updated = bstate.update_if_better(cstate.x, cstate.f);

        if (updated)
        {
            // to keep the gradient up to date as well
            bstate.g = cstate.g;
        }

        if (updated && df >= epsilon)
        {
            last_ibest = i;
        }
        else if (i >= patience + last_ibest)
        {
            // decrease step length ratio if not significant decrease recently
            h /= gamma;
            last_ibest = i;
            cstate = bstate;
        }

        const auto converged = (cstate.g.lpNorm<2>() < epsilon0<scalar_t>()) || ((h <= L * epsilon) && (df < epsilon));
        if (solver_t::done(function, bstate, iter_ok, converged))
        {
            break;
        }

        // update estimation of the Lipschitz constant
        L = std::max(L, cstate.g.lpNorm<2>());
    }

    return bstate;
}
