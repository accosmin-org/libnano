#include <nano/lsearchk/backtrack.h>

using namespace nano;

lsearchk_backtrack_t::lsearchk_backtrack_t()
    : lsearchk_t("backtrack")
{
    type(lsearch_type::armijo);
    register_parameter(parameter_t::make_enum("lsearchk::backtrack::interpolation", interpolation_type::cubic));
    register_parameter(parameter_t::make_scalar("lsearchk::backtrack::safeguard", 0.0, LT, 0.1, LT, 0.5));
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
    const auto safeguard      = parameter("lsearchk::backtrack::safeguard").value<scalar_t>();

    for (int i = 0; i < max_iterations && state.valid(); ++i)
    {
        if (state.has_armijo(state0, c1))
        {
            return true;
        }

        // next trial
        const auto tmin       = std::min(state0.t, state.t);
        const auto tmax       = std::max(state0.t, state.t);
        const auto interp_min = tmin + safeguard * (tmax - tmin);
        const auto interp_max = tmax - safeguard * (tmax - tmin);
        const auto next       = lsearch_step_t::interpolate(state0, state, interpolation);
        const auto ok         = state.update(state0, std::clamp(next, interp_min, interp_max));
        log(state0, state);

        if (!ok)
        {
            return false;
        }
    }

    return false;
}
