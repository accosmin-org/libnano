#include <nano/core/numeric.h>
#include <nano/lsearchk/cgdescent.h>

#include <iomanip>
#include <iostream>

using namespace nano;

static auto initial_params(const estimator_t& estimator)
{
    const auto [c1, c2]       = estimator.parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = estimator.parameter("lsearchk::max_iterations").value<int>();
    const auto theta          = estimator.parameter("lsearchk::cgdescent::theta").value<scalar_t>();
    const auto epsilon        = estimator.parameter("lsearchk::cgdescent::epsilon").value<scalar_t>();
    const auto ro             = estimator.parameter("lsearchk::cgdescent::ro").value<scalar_t>();
    const auto omega          = estimator.parameter("lsearchk::cgdescent::omega").value<scalar_t>();
    const auto delta          = estimator.parameter("lsearchk::cgdescent::delta").value<scalar_t>();
    const auto gamma          = estimator.parameter("lsearchk::cgdescent::gamma").value<scalar_t>();
    const auto criterion      = estimator.parameter("lsearchk::cgdescent::criterion").value<cgdescent_criterion_type>();

    return std::make_tuple(c1, c2, max_iterations, theta, epsilon, ro, omega, delta, gamma, criterion);
}

lsearchk_cgdescent_t::lsearchk_cgdescent_t()
{
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::epsilon", 0, LT, 1e-6, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::theta", 0, LT, 0.5, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::gamma", 0, LT, 0.66, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::delta", 0, LT, 0.7, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::omega", 0, LT, 1e-3, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::ro", 1, LT, 5.0, LT, 1e+6));
    register_parameter(
        parameter_t::make_enum("lsearchk::cgdescent::criterion", cgdescent_criterion_type::wolfe_approx_wolfe));
}

rlsearchk_t lsearchk_cgdescent_t::clone() const
{
    auto lsearchk      = std::make_unique<lsearchk_cgdescent_t>(*this);
    lsearchk->m_Qk     = 0;
    lsearchk->m_Ck     = 0;
    lsearchk->m_approx = false;
    return rlsearchk_t{lsearchk.release()};
}

scalar_t lsearchk_cgdescent_t::approx_armijo_epsilon() const
{
    return parameter("lsearchk::cgdescent::epsilon").value<scalar_t>() * m_Ck;
}

bool lsearchk_cgdescent_t::done(const state_t& state, const cgdescent_criterion_type criterion, const scalar_t c1,
                                const scalar_t c2, const scalar_t epsilon, const scalar_t omega,
                                [[maybe_unused]] const bool bracketed)
{
    assert(state.a.t <= state.b.t);
    assert(state.a.g < 0.0);

    assert(!bracketed || state.a.f <= state.state0.f + epsilon * m_Ck);
    assert(!bracketed || state.b.g >= 0.0);

    if (!static_cast<bool>(state.c))
    {
        // diverged
        return true;
    }

    if (state.c.t < state.a.t || state.c.t > state.b.t)
    {
        // tentative point outside bracketing interval, keep trying
        return false;
    }

    switch (criterion)
    {
    case cgdescent_criterion_type::wolfe: return state.has_wolfe(c1, c2);

    case cgdescent_criterion_type::approx_wolfe: return state.has_approx_wolfe(c1, c2, epsilon * m_Ck);

    default:
        if (!m_approx)
        {
            if (state.has_wolfe(c1, c2))
            {
                // decide if to switch permanently to the approximate Wolfe conditions
                m_approx = std::fabs(state.c.f - state.state0.f) <= omega * m_Ck;
                return true;
            }
            return false;
        }
        else
        {
            return state.has_approx_wolfe(c1, c2, epsilon * m_Ck);
        }
    }
}

void lsearchk_cgdescent_t::move(state_t& state, const scalar_t t) const
{
    state.c.update(state.state0, t);
    log(state.state0, state.c);
}

void lsearchk_cgdescent_t::updateU(state_t& state, const scalar_t epsilon, const scalar_t theta,
                                   const int max_iterations) const
{
    auto& a = state.a;
    auto& b = state.b;
    auto& d = state.c;

    for (int i = 0; i < max_iterations && (b.t - a.t) > stpmin(); ++i)
    {
        move(state, (1 - theta) * a.t + theta * b.t);
        if (!d)
        {
            return;
        }
        else if (!d.has_descent())
        {
            b = d;
            return;
        }
        else if (d.has_approx_armijo(state.state0, epsilon * m_Ck))
        {
            a = d;
        }
        else
        {
            b = d;
        }
    }
}

void lsearchk_cgdescent_t::update(state_t& state, const scalar_t epsilon, const scalar_t theta,
                                  const int max_iterations) const
{
    auto& a = state.a;
    auto& b = state.b;
    auto& c = state.c;

    if (c.t <= a.t || c.t >= b.t)
    {
        return;
    }
    else if (!c.has_descent())
    {
        b = c;
    }
    else if (c.has_approx_armijo(state.state0, epsilon * m_Ck))
    {
        a = c;
    }
    else
    {
        b = c;
        updateU(state, epsilon, theta, max_iterations);
    }
}

void lsearchk_cgdescent_t::bracket(state_t& state, const scalar_t ro, const scalar_t epsilon, const scalar_t theta,
                                   const int max_iterations) const
{
    auto& a      = state.a;
    auto& b      = state.b;
    auto& c      = state.c;
    auto  last_a = a;

    for (int i = 0; i < max_iterations && static_cast<bool>(c); ++i)
    {
        if (!c.has_descent())
        {
            a = last_a;
            b = c;
            return;
        }
        else if (!c.has_approx_armijo(state.state0, epsilon * m_Ck))
        {
            a = state.state0;
            b = c;
            updateU(state, epsilon, theta, max_iterations);
            return;
        }
        else
        {
            last_a = c;
            move(state, ro * c.t);
        }
    }
}

bool lsearchk_cgdescent_t::get(const solver_state_t& state0, solver_state_t& state)
{
    assert(state0.has_descent());

    const auto [c1, c2, max_iterations, theta, epsilon, ro, omega, delta, gamma, criterion] = initial_params(*this);

    // estimate an upper bound of the function value
    // (to be used for the approximate Wolfe condition)
    m_Qk = 1.0 + m_Qk * delta;
    m_Ck = m_Ck + (std::fabs(state0.f) - m_Ck) / m_Qk;

    std::cout << "t0=" << state0.t << ",t=" << state.t << std::endl;

    /*
    const auto epsilonk =
        (criterion == cgdescent_criterion_type::wolfe ||
         (criterion == cgdescent_criterion_type::wolfe_approx_wolfe && !m_approx)) ?
        c1 : (epsilon * m_Ck);
    */

    // current bracketing interval
    auto interval = state_t{state0, state};
    if (done(interval, criterion, c1, c2, epsilon, omega, false))
    {
        return static_cast<bool>(state);
    }

    // bracket the initial step size
    bracket(interval, ro, epsilon, theta, max_iterations);
    if (done(interval, criterion, c1, c2, epsilon, omega))
    {
        return static_cast<bool>(state);
    }

    const auto move_update_and_check_done = [&](const auto t)
    {
        move(interval, t);
        if (done(interval, criterion, c1, c2, epsilon, omega))
        {
            return true;
        }

        update(interval, epsilon, theta, max_iterations);
        return done(interval, criterion, c1, c2, epsilon, omega);
    };

    // iteratively update the search interval [a, b]
    for (int i = 0; i < max_iterations && (interval.b.t - interval.a.t) > stpmin(); ++i)
    {
        auto& a = interval.a;
        auto& b = interval.b;

        std::cout << std::fixed << std::setprecision(12) << "i=" << i << ", a=[t=" << a.t << ",f=" << a.f
                  << ",a=" << (a.f <= state0.f + c1 * a.t * state0.dg()) << ",w=" << (a.g >= c2 * state0.dg()) << "]"
                  << ",b=[t=" << b.t << ",f=" << b.f << ",a=" << (b.f <= state0.f + c1 * b.t * state0.dg())
                  << ",w=" << (b.g >= c2 * state0.dg()) << "], c=[t=" << interval.c.t << "]\n";

        // secant interpolation
        const auto prev_width = b.t - a.t;
        const auto a0 = a, b0 = b;
        const auto tc = lsearch_step_t::secant(a0, b0);

        if (move_update_and_check_done(tc))
        {
            return static_cast<bool>(state);
        }
        else if (std::fabs(tc - a.t) < epsilon0<scalar_t>())
        {
            if (move_update_and_check_done(lsearch_step_t::secant(a0, a)))
            {
                return static_cast<bool>(state);
            }
        }
        else if (std::fabs(tc - b.t) < epsilon0<scalar_t>())
        {
            if (move_update_and_check_done(lsearch_step_t::secant(b0, b)))
            {
                return static_cast<bool>(state);
            }
        }

        // update search interval
        if (b.t - a.t > gamma * prev_width)
        {
            if (move_update_and_check_done((a.t + b.t) / 2))
            {
                return static_cast<bool>(state);
            }
        }
    }

    return false;
}
