#include <nano/solver/asgm.h>

using namespace nano;

solver_asgm_t::solver_asgm_t()
{
    monotonic(false);
}

solver_state_t solver_asgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    auto function = make_function(function_, x0);

    auto cstate = solver_state_t{function, x0}; // current state
    auto bstate = cstate;                       // best state

    const auto epsilon = this->epsilon();

    const auto gamma = m_gamma.get();
    const auto patience = m_patience.get();

    auto h = 1.0;                               // current step length ratio
    auto L = cstate.g.lpNorm<2>();              // estimation of the Lipschitz constant

    for (int64_t i = 0, last_ibest = 0; function.fcalls() < max_evals(); ++ i)
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

        const auto converged = (h <= L * epsilon) && (df < epsilon);
        if (solver_t::done(function, bstate, iter_ok, converged))
        {
            break;
        }

        // update estimation of the Lipschitz constant
        L = std::max(L, cstate.g.lpNorm<2>());
    }

    return bstate;
}
