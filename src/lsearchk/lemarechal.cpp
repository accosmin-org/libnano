#include <nano/core/numeric.h>
#include <nano/lsearchk/lemarechal.h>

using namespace nano;

lsearchk_lemarechal_t::lsearchk_lemarechal_t()
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

bool lsearchk_lemarechal_t::get(const solver_state_t& state0, solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto tau1           = parameter("lsearchk::lemarechal::tau1").value<scalar_t>();
    const auto interpolation  = parameter("lsearchk::lemarechal::interpolation").value<interpolation_type>();
    const auto safeguard      = parameter("lsearchk::lemarechal::safeguard").value<scalar_t>();

    lsearch_step_t L = state0;
    lsearch_step_t R = L;

    bool R_updated = false;
    for (int i = 1; i < max_iterations; ++i)
    {
        scalar_t tmin = 0, tmax = 0;
        if (state.has_armijo(state0, c1))
        {
            if (state.has_wolfe(state0, c2))
            {
                return true;
            }
            else
            {
                L = state;
                if (!R_updated)
                {
                    tmin = std::max(L.t, R.t) + 2 * std::fabs(L.t - R.t);
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
            R         = state;
            R_updated = true;
            tmin      = std::min(L.t, R.t);
            tmax      = std::max(L.t, R.t);
        }

        // next trial
        const auto interp_min = tmin + safeguard * (tmax - tmin);
        const auto interp_max = tmax - safeguard * (tmax - tmin);
        const auto next       = lsearch_step_t::interpolate(L, R, interpolation);
        const auto ok         = state.update(state0, std::clamp(next, interp_min, interp_max));
        log(state0, state);

        if (!ok)
        {
            return false;
        }
    }

    return false;
}
