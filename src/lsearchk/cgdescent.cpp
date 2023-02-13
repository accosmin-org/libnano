#include <nano/core/numeric.h>
#include <nano/lsearchk/cgdescent.h>

using namespace nano;

static auto initial_params(const configurable_t& configurable)
{
    const auto [c1, c2]       = configurable.parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = configurable.parameter("lsearchk::max_iterations").value<int>();
    const auto theta          = configurable.parameter("lsearchk::cgdescent::theta").value<scalar_t>();
    const auto epsilon        = configurable.parameter("lsearchk::cgdescent::epsilon").value<scalar_t>();
    const auto ro             = configurable.parameter("lsearchk::cgdescent::ro").value<scalar_t>();
    const auto gamma          = configurable.parameter("lsearchk::cgdescent::gamma").value<scalar_t>();

    return std::make_tuple(c1, c2, max_iterations, theta, epsilon, ro, gamma);
}

lsearchk_cgdescent_t::lsearchk_cgdescent_t()
    : lsearchk_t("cgdescent")
{
    type(lsearch_type::wolfe_approx_wolfe);
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::epsilon", 0, LT, 1e-6, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::theta", 0, LT, 0.5, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::gamma", 0, LT, 0.66, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::ro", 1, LT, 5.0, LT, 1e+6));
}

rlsearchk_t lsearchk_cgdescent_t::clone() const
{
    return std::make_unique<lsearchk_cgdescent_t>(*this);
}

bool lsearchk_cgdescent_t::done(const state_t& state, const scalar_t c1, const scalar_t c2, const scalar_t epsilonk,
                                [[maybe_unused]] const bool bracketed)
{
    assert(state.a.t <= state.b.t);
    assert(state.a.g < 0.0);

    if ((bracketed && (state.a.f > state.state0.f + epsilonk || state.b.g < 0.0)) || !state.c.valid())
    {
        // bracketing failed or diverged
        return true;
    }
    else if (state.c.t < state.a.t || state.c.t > state.b.t)
    {
        // tentative point outside bracketing interval, keep trying
        return false;
    }
    else
    {
        return
            // (Armijo-)Wolfe conditions
            (state.c.has_armijo(state.state0, c1) && state.c.has_wolfe(state.state0, c2)) ||

            // approximate (Armijo-)Wolfe conditions
            (state.c.has_approx_armijo(state.state0, epsilonk) && state.c.has_approx_wolfe(state.state0, c1, c2));
    }
}

void lsearchk_cgdescent_t::move(state_t& state, const scalar_t t) const
{
    state.c.update(state.state0, t);
    log(state.state0, state.c);
}

void lsearchk_cgdescent_t::updateU(state_t& state, const scalar_t epsilonk, const scalar_t theta,
                                   const int max_iterations) const
{
    for (int i = 0; i < max_iterations && (state.b.t - state.a.t) > stpmin(); ++i)
    {
        const auto& d = state.c;
        move(state, (1 - theta) * state.a.t + theta * state.b.t);

        if (!d.valid())
        {
            return;
        }
        else if (!d.has_descent())
        {
            state.b = d;
            return;
        }
        else if (d.has_approx_armijo(state.state0, epsilonk))
        {
            state.a = d;
        }
        else
        {
            state.b = d;
        }
    }
}

void lsearchk_cgdescent_t::update(state_t& state, const scalar_t epsilonk, const scalar_t theta,
                                  const int max_iterations) const
{
    if (state.c.t <= state.a.t || state.c.t >= state.b.t)
    {
        return;
    }
    else if (!state.c.has_descent())
    {
        state.b = state.c;
    }
    else if (state.c.has_approx_armijo(state.state0, epsilonk))
    {
        state.a = state.c;
    }
    else
    {
        state.b = state.c;
        updateU(state, epsilonk, theta, max_iterations);
    }
}

void lsearchk_cgdescent_t::bracket(state_t& state, const scalar_t ro, const scalar_t epsilonk, const scalar_t theta,
                                   const int max_iterations) const
{
    auto last_a = state.a;

    for (int i = 0; i < max_iterations && state.c.valid(); ++i)
    {
        if (!state.c.has_descent())
        {
            state.a = last_a;
            state.b = state.c;
            return;
        }
        else if (!state.c.has_approx_armijo(state.state0, epsilonk))
        {
            state.a = state.state0;
            state.b = state.c;
            updateU(state, epsilonk, theta, max_iterations);
            return;
        }
        else
        {
            last_a = state.c;
            move(state, ro * state.c.t);
        }
    }
}

bool lsearchk_cgdescent_t::get(const solver_state_t& state0, solver_state_t& state) const
{
    assert(state0.has_descent());

    const auto [c1, c2, max_iterations, theta, epsilon, ro, gamma] = initial_params(*this);

    const auto epsilonk = epsilon * std::fabs(state0.f);

    // current bracketing interval
    assert(state0.has_descent());
    auto interval = state_t{state0, state};
    if (done(interval, c1, c2, epsilonk, false))
    {
        return state.valid();
    }

    // bracket the initial step size
    bracket(interval, ro, epsilonk, theta, max_iterations);
    if (done(interval, c1, c2, epsilonk))
    {
        return state.valid();
    }

    const auto move_update_and_check_done =
        [&, c1 = c1, c2 = c2, theta = theta, max_iterations = max_iterations](const auto t)
    {
        if (!std::isfinite(t))
        {
            // interpolation failed, go on with bisection?!
            return false;
        }

        move(interval, t);
        if (done(interval, c1, c2, epsilonk))
        {
            return true;
        }

        update(interval, epsilonk, theta, max_iterations);
        return done(interval, c1, c2, epsilonk);
    };

    // iteratively update the search interval [a, b]
    for (int i = 0; i < max_iterations && (interval.b.t - interval.a.t) > stpmin(); ++i)
    {
        const auto& a = interval.a;
        const auto& b = interval.b;

        // secant interpolation
        const auto prev_width = b.t - a.t;
        const auto a0 = a, b0 = b;
        const auto tc = lsearch_step_t::secant(a0, b0);

        if (move_update_and_check_done(tc))
        {
            return state.valid();
        }
        else if (std::fabs(tc - a.t) < epsilon0<scalar_t>())
        {
            if (move_update_and_check_done(lsearch_step_t::secant(a0, a)))
            {
                return state.valid();
            }
        }
        else if (std::fabs(tc - b.t) < epsilon0<scalar_t>())
        {
            if (move_update_and_check_done(lsearch_step_t::secant(b0, b)))
            {
                return state.valid();
            }
        }

        // update search interval
        if (b.t - a.t > gamma * prev_width)
        {
            if (move_update_and_check_done((a.t + b.t) / 2))
            {
                return state.valid();
            }
        }
    }

    return false;
}
