#include <nano/lsearchk/backtrack.h>

using namespace nano;

lsearchk_backtrack_t::lsearchk_backtrack_t()
{
    register_parameter(parameter_t::make_enum("lsearchk::backtrack::interpolation", interpolation_type::cubic));
}

rlsearchk_t lsearchk_backtrack_t::clone() const
{
    return std::make_unique<lsearchk_backtrack_t>(*this);
}

bool lsearchk_backtrack_t::get(const solver_state_t& state0, solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto interpolation  = parameter("lsearchk::backtrack::interpolation").value<interpolation_type>();

    for (int i = 0; i < max_iterations && state; ++i)
    {
        if (state.has_armijo(state0, c1))
        {
            return true;
        }

        // next trial
        state.update(state0, lsearch_step_t::interpolate(state0, state, interpolation));
        log(state0, state);
    }

    return false;
}
