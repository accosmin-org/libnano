#include <nano/lsearchk/backtrack.h>

using namespace nano;

rlsearchk_t lsearchk_backtrack_t::clone() const
{
    return std::make_unique<lsearchk_backtrack_t>(*this);
}

bool lsearchk_backtrack_t::get(const solver_state_t& state0, solver_state_t& state)
{
    for (int64_t i = 0; i < max_iterations() && state; ++ i)
    {
        if (state.has_armijo(state0, c1()))
        {
            return true;
        }

        // next trial
        state.update(state0, lsearch_step_t::interpolate(state0, state, m_interpolation));
        log(state0, state);
    }

    return false;
}
