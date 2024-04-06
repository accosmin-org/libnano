#include <nano/core/numeric.h>
#include <nano/lsearchk/lemarechal.h>

using namespace nano;

lsearchk_lemarechal_t::lsearchk_lemarechal_t()
    : lsearchk_t("lemarechal")
{
    type(lsearch_type::wolfe);
    register_parameter(parameter_t::make_enum("lsearchk::lemarechal::interpolation", interpolation_type::cubic));
    register_parameter(parameter_t::make_scalar("lsearchk::lemarechal::tau1", 2, LT, 9, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("lsearchk::lemarechal::safeguard", 0.0, LT, 0.1, LT, 0.5));
}

rlsearchk_t lsearchk_lemarechal_t::clone() const
{
    return std::make_unique<lsearchk_lemarechal_t>(*this);
}

lsearchk_t::result_t lsearchk_lemarechal_t::do_get(const solver_state_t& state0, const vector_t& descent,
                                                   scalar_t step_size, solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto tau1           = parameter("lsearchk::lemarechal::tau1").value<scalar_t>();
    const auto interpolation  = parameter("lsearchk::lemarechal::interpolation").value<interpolation_type>();
    const auto safeguard      = parameter("lsearchk::lemarechal::safeguard").value<scalar_t>();

    auto L = lsearch_step_t{state0, descent, 0.0};
    auto R = L;

    bool R_updated = false;
    for (int i = 1; i < max_iterations; ++i)
    {
        scalar_t tmin = 0;
        scalar_t tmax = 0;
        if (state.has_armijo(state0, descent, step_size, c1))
        {
            if (state.has_wolfe(state0, descent, c2))
            {
                return {true, step_size};
            }
            else
            {
                L = {state, descent, step_size};
                if (!R_updated)
                {
                    tmin = std::max(L.t, R.t) + 1.0 * std::fabs(L.t - R.t);
                    tmax = std::max(L.t, R.t) + tau1 * std::fabs(L.t - R.t);
                }
                else
                {
                    tmin = std::min(L.t, R.t);
                    tmax = std::max(L.t, R.t);
                }
            }
        }
        else
        {
            R         = {state, descent, step_size};
            R_updated = true;
            tmin      = std::min(L.t, R.t);
            tmax      = std::max(L.t, R.t);
        }

        // next trial
        const auto interp_min = tmin + safeguard * (tmax - tmin);
        const auto interp_max = tmax - safeguard * (tmax - tmin);

        step_size = lsearch_step_t::interpolate(L, R, interpolation);
        step_size = std::clamp(step_size, interp_min, interp_max);
        if (!update(state, state0, descent, step_size))
        {
            return {false, step_size};
        }
    }

    return {false, step_size};
}
