#include <solver/bundle/csearch.h>
#include <solver/bundle/proximal.h>
#include <solver/rqb.h>

using namespace nano;

solver_rqb_t::solver_rqb_t()
    : solver_t("rqb")
{
    const auto prefix = string_t{"solver::rqb"};
    bundle_t::config(*this, prefix);
    csearch_t::config(*this, prefix);
    proximal_t::config(*this, prefix);
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
    auto bundle   = bundle_t::make(state, *this, prefix);
    auto csearch  = csearch_t::make(function, *this, prefix);
    auto proximal = proximal_t::make(state, *this, prefix);

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        const auto& cpoint = csearch.search(bundle, proximal.miu(), max_evals, epsilon, logger);
        [[maybe_unused]] const auto& [t, status, y, gy, fy, ghat, fhat] = cpoint;

        const auto iter_ok   = status != csearch_status::failed;
        const auto converged = status == csearch_status::converged;
        if (solver_t::done_specific_test(state, iter_ok, converged, logger))
        {
            break;
        }

        if (status == csearch_status::descent_step || status == csearch_status::cutting_plane_step)
        {
            assert(fy < state.fx());

            proximal.update(bundle, cpoint);
            bundle.moveto(y, gy, fy);
            state.update(y, gy, fy);
        }
        else if (status == csearch_status::null_step)
        {
            proximal.update(bundle, cpoint);
            bundle.append(y, gy, fy);
        }
    }

    state.update_calls();
    return state;
}
