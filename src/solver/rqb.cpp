#include <nano/solver/csearch.h>
#include <nano/solver/proximity.h>
#include <nano/solver/rqb.h>

using namespace nano;

solver_rqb_t::solver_rqb_t()
    : solver_t("rqb")
{
    type(solver_type::non_monotonic);

    const auto prefix = string_t{"solver::rqb"};
    bundle_t::config(*this, prefix);
    csearch_t::config(*this, prefix);
    proximity_t::config(*this, prefix);
}

rsolver_t solver_rqb_t::clone() const
{
    return std::make_unique<solver_rqb_t>(*this);
}

solver_state_t solver_rqb_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto prefix    = string_t{"solver::rqb"};
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();

    auto state     = solver_state_t{function, x0};
    auto bundle    = bundle_t::make(state, *this, prefix);
    auto csearch   = csearch_t::make(function, *this, prefix);
    auto proximity = proximity_t::make(state, *this, prefix);

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        const auto& [t, status, y, gy, fy] = csearch.search(bundle, proximity.miu(), max_evals, epsilon);

        const auto iter_ok   = status != csearch_status::failed;
        const auto converged = status == csearch_status::converged;
        if (solver_t::done(state, iter_ok, converged))
        {
            break;
        }

        if (status == csearch_status::descent_step)
        {
            proximity.update(t, bundle.x(), y, bundle.gx(), gy);
            bundle.moveto(y, gy, fy);
            assert(fy < state.fx());
            state.update(y, gy, fy);
        }
        else if (status == csearch_status::cutting_plane_step)
        {
            bundle.moveto(y, gy, fy);
            assert(fy < state.fx());
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
