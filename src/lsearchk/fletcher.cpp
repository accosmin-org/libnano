#include <nano/core/numeric.h>
#include <nano/lsearchk/fletcher.h>

using namespace nano;

lsearchk_fletcher_t::lsearchk_fletcher_t()
{
    type(lsearch_type::strong_wolfe);
    register_parameter(parameter_t::make_enum("lsearchk::fletcher::interpolation", interpolation_type::cubic));
    register_parameter(parameter_t::make_scalar("lsearchk::fletcher::tau1", 2, LT, 9.0, LT, 1e+6));
    register_parameter(parameter_t::make_scalar_pair("lsearchk::fletcher::tau23", 0, LT, 0.1, LT, 0.5, LE, 0.5));
}

rlsearchk_t lsearchk_fletcher_t::clone() const
{
    return std::make_unique<lsearchk_fletcher_t>(*this);
}

bool lsearchk_fletcher_t::zoom(const solver_state_t& state0, lsearch_step_t lo, lsearch_step_t hi,
                               solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto [tau2, tau3]   = parameter("lsearchk::fletcher::tau23").value_pair<scalar_t>();
    const auto interpolation  = parameter("lsearchk::fletcher::interpolation").value<interpolation_type>();

    for (int i = 0; i < max_iterations && std::fabs(lo.t - hi.t) > epsilon0<scalar_t>(); ++i)
    {
        const auto tmin = lo.t + std::min(tau2, c2) * (hi.t - lo.t);
        const auto tmax = hi.t - tau3 * (hi.t - lo.t);
        const auto next = lsearch_step_t::interpolate(lo, hi, interpolation);
        const auto ok   = state.update(state0, std::clamp(next, std::min(tmin, tmax), std::max(tmin, tmax)));
        log(state0, state);

        if (!ok)
        {
            return false;
        }
        else if (!state.has_armijo(state0, c1) || state.f >= lo.f)
        {
            hi = state;
        }
        else
        {
            if (state.has_strong_wolfe(state0, c2))
            {
                return true;
            }

            if (state.dg() * (hi.t - lo.t) >= scalar_t(0))
            {
                hi = lo;
            }
            lo = state;
        }
    }

    return false;
}

bool lsearchk_fletcher_t::get(const solver_state_t& state0, solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto tau1           = parameter("lsearchk::fletcher::tau1").value<scalar_t>();
    const auto interpolation  = parameter("lsearchk::fletcher::interpolation").value<interpolation_type>();

    lsearch_step_t prev = state0;
    lsearch_step_t curr = state;

    for (int i = 1; i < max_iterations; ++i)
    {
        if (!state.has_armijo(state0, c1) || (state.f >= prev.f && i > 1))
        {
            return zoom(state0, prev, curr, state);
        }
        else if (state.has_strong_wolfe(state0, c2))
        {
            return true;
        }
        else if (!state.has_descent())
        {
            return zoom(state0, curr, prev, state);
        }

        // next trial
        const auto tmin = curr.t + 2 * (curr.t - prev.t);
        const auto tmax = curr.t + tau1 * (curr.t - prev.t);
        const auto next = lsearch_step_t::interpolate(prev, curr, interpolation);
        const auto ok   = state.update(state0, std::clamp(next, tmin, tmax));
        log(state0, state);

        if (!ok)
        {
            return false;
        }
        prev = curr;
        curr = state;
    }

    return false;
}
