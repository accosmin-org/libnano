#include <solver/bundle/csearch.h>
#include <solver/bundle/quasi.h>
#include <solver/rqb.h>

using namespace nano;
using namespace nano::bundle;

solver_rqb_t::solver_rqb_t()
    : solver_t("rqb")
{
    const auto prefix = string_t{"solver::rqb"};
    quasi_t::config(*this, prefix);
    bundle_t::config(*this, prefix);
    csearch_t::config(*this, prefix);
}

rsolver_t solver_rqb_t::clone() const
{
    return std::make_unique<solver_rqb_t>(*this);
}

solver_state_t solver_rqb_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    warn_nonconvex(function, logger);
    warn_constrained(function, logger);

    const auto prefix    = string_t{"solver::rqb"};
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();

    auto state    = solver_state_t{function, x0};
    auto quasi    = quasi_t::make(state, *this, prefix);
    auto bundle   = bundle_t::make(state, *this, prefix);
    auto csearch  = csearch_t::make(function, *this, prefix);

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        const auto& cpoint = csearch.search(bundle, quasi.M(), max_evals, epsilon, logger);
        [[maybe_unused]] const auto& [t, status, y, gy, fy, ghat, fhat] = cpoint;

        const auto iter_ok   = status != csearch_status::failed;
        const auto converged = status == csearch_status::converged;
        if (solver_t::done_specific_test(state, iter_ok, converged, logger))
        {
            break;
        }

        // FIXME: descent step and cutting plane step are the same ?!
        // FIXME: the proximal parameter is updated from time to time

        if (status == csearch_status::descent_step)
        {
            quasi.update(y, gy, ghat, true);
            bundle.moveto(y, gy, fy);
            state.update(y, gy, fy);
        }
        else if (status == csearch_status::cutting_plane_step)
        {
            quasi.update(y, gy, ghat, false);
            bundle.moveto(y, gy, fy);
            state.update(y, gy, fy);
        }
        else if (status == csearch_status::null_step)
        {
            bundle.append(y, gy, fy);
        }
    }

    state.update_calls();
    return state;
}
