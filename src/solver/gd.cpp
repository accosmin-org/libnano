#include <nano/solver/gd.h>

using namespace nano;

solver_gd_t::solver_gd_t()
    : solver_t("gd")
{
    monotonic(true);
    parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1);
}

rsolver_t solver_gd_t::clone() const
{
    return std::make_unique<solver_gd_t>(*this);
}

solver_state_t solver_gd_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();

    auto lsearch = make_lsearch();

    auto cstate = solver_state_t{function, x0};
    if (solver_t::done(function, cstate, true, cstate.converged(epsilon)))
    {
        return cstate;
    }

    for (int64_t i = 0; function.fcalls() < max_evals; ++i)
    {
        // descent direction
        cstate.d = -cstate.g;

        // line-search
        const auto iter_ok = lsearch.get(cstate);
        if (solver_t::done(function, cstate, iter_ok, cstate.converged(epsilon)))
        {
            break;
        }
    }

    return cstate;
} // LCOV_EXCL_LINE
