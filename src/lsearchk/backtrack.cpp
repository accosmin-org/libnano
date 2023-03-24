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

lsearchk_t::result_t lsearchk_backtrack_t::do_get(const solver_state_t& state0, const vector_t& descent,
                                                  scalar_t step_size, solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto interpolation  = parameter("lsearchk::backtrack::interpolation").value<interpolation_type>();
    const auto safeguard      = parameter("lsearchk::backtrack::safeguard").value<scalar_t>();

    for (int i = 0; i < max_iterations && state.valid(); ++i)
    {
        if (state.has_armijo(state0, descent, step_size, c1))
        {
            return {true, step_size};
        }

        // next trial
        const auto tmin       = std::min(0.0, step_size);
        const auto tmax       = std::max(0.0, step_size);
        const auto interp_min = tmin + safeguard * (tmax - tmin);
        const auto interp_max = tmax - safeguard * (tmax - tmin);

        const auto step0 = lsearch_step_t{state0, descent, 0.0};
        const auto step  = lsearch_step_t{state, descent, step_size};

        step_size = lsearch_step_t::interpolate(step0, step, interpolation);
        step_size = std::clamp(step_size, interp_min, interp_max);

        if (!update(state, state0, descent, step_size))
        {
            return {false, step_size};
        }
    }

    return {false, step_size};
}
