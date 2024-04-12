#include <nano/core/numeric.h>
#include <nano/lsearchk/fletcher.h>

using namespace nano;

lsearchk_fletcher_t::lsearchk_fletcher_t()
    : lsearchk_t("fletcher")
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

lsearchk_t::result_t lsearchk_fletcher_t::zoom(const solver_state_t& state0, const vector_t& descent, lsearch_step_t lo,
                                               lsearch_step_t hi, solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto [tau2, tau3]   = parameter("lsearchk::fletcher::tau23").value_pair<scalar_t>();
    const auto interpolation  = parameter("lsearchk::fletcher::interpolation").value<interpolation_type>();

    for (int i = 0; i < max_iterations && std::fabs(lo.t - hi.t) > epsilon0<scalar_t>(); ++i)
    {
        const auto tmin = std::min(lo.t, hi.t) + std::min(tau2, c2) * std::fabs(hi.t - lo.t);
        const auto tmax = std::max(lo.t, hi.t) - tau3 * std::fabs(hi.t - lo.t);
        assert(tmin < tmax);

        auto step_size = lsearch_step_t::interpolate(lo, hi, interpolation);
        step_size      = std::clamp(step_size, tmin, tmax);

        if (!update(state, state0, descent, step_size))
        {
            return {false, step_size};
        }
        else if (!state.has_armijo(state0, descent, step_size, c1) || state.fx() >= lo.f)
        {
            hi = {state, descent, step_size};
        }
        else if (state.has_strong_wolfe(state0, descent, c2))
        {
            return {true, step_size};
        }
        else
        {
            if (state.dg(descent) * (hi.t - lo.t) >= 0.0)
            {
                hi = lo;
            }
            lo = {state, descent, step_size};
        }
    }

    return {false, hi.t};
}

lsearchk_t::result_t lsearchk_fletcher_t::do_get(const solver_state_t& state0, const vector_t& descent,
                                                 scalar_t step_size, solver_state_t& state) const
{
    const auto [c1, c2]       = parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();
    const auto tau1           = parameter("lsearchk::fletcher::tau1").value<scalar_t>();
    const auto interpolation  = parameter("lsearchk::fletcher::interpolation").value<interpolation_type>();

    auto prev = lsearch_step_t{state0, descent, 0.0};
    auto curr = lsearch_step_t{state, descent, step_size};

    for (int i = 1; i < max_iterations; ++i)
    {
        assert(prev.t < curr.t);

        if (!state.has_armijo(state0, descent, step_size, c1) || curr.f >= prev.f)
        {
            return zoom(state0, descent, prev, curr, state);
        }
        else if (state.has_strong_wolfe(state0, descent, c2))
        {
            return {true, step_size};
        }
        else if (!state.has_descent(descent))
        {
            return zoom(state0, descent, curr, prev, state);
        }

        // next trial
        const auto tmin = curr.t + 1.0 * (curr.t - prev.t);
        const auto tmax = curr.t + tau1 * (curr.t - prev.t);
        assert(tmin < tmax);

        step_size = lsearch_step_t::interpolate(prev, curr, interpolation);
        step_size = std::clamp(step_size, tmin, tmax);

        if (!update(state, state0, descent, step_size))
        {
            return {false, step_size};
        }
        prev = curr;
        curr = {state, descent, step_size};
    }

    return {false, step_size};
}
