#include "cgdescent.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearchk_cgdescent_t::config() const
{
    json_t json;
    json["epsilon"] = strcat(m_epsilon, "(0,inf)");
    json["theta"] = strcat(m_theta, "(0,1)");
    json["gamma"] = strcat(m_gamma, "(0,1)");
    json["delta"] = strcat(m_delta, "[0,1]");
    json["omega"] = strcat(m_omega, "[0,1]");
    json["ro"] = strcat(m_ro, "(1,inf)");
    return json;
}

void lsearchk_cgdescent_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "epsilon", m_epsilon, eps, inf);
    nano::from_json_range(json, "theta", m_theta, eps, 1 - eps);
    nano::from_json_range(json, "gamma", m_gamma, eps, 1 - eps);
    nano::from_json_range(json, "delta", m_delta, 0, 1);
    nano::from_json_range(json, "omega", m_omega, 0, 1);
    nano::from_json_range(json, "ro", m_ro, 1 + eps, inf);
}

bool lsearchk_cgdescent_t::updateU(const solver_state_t& state0, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    for (int i = 0; i < max_iterations(); ++ i)
    {
        if (evaluate(state0, (1 - m_theta) * a.t + m_theta * b.t, c))
        {
            return true;
        }
        else if (!c.has_descent())
        {
            b = c;
            return false;
        }
        else if (c.has_approx_armijo(state0, m_epsilon * m_sumC))
        {
            a = c;
        }
        else
        {
            b = c;
        }
    }

    return false;
}

bool lsearchk_cgdescent_t::update(const solver_state_t& state0, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    if (c.t <= a.t || c.t >= b.t)
    {
        return false;
    }
    else if (!c.has_descent())
    {
        b = c;
        return false;
    }
    else if (c.has_approx_armijo(state0, m_epsilon * m_sumC))
    {
        a = c;
        return false;
    }
    else
    {
        b = c;
        return updateU(state0, a, b, c);
    }
}

bool lsearchk_cgdescent_t::secant2(const solver_state_t& state0, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    // NB: using safeguarded cubic interpolation instead of the secant interpolation
    //  because it converges faster and it is not numerically robust!

    const auto a0 = a, b0 = b;
    const auto tc = lsearch_step_t::interpolate(a0, b0);

    if (evaluate(state0, tc, c))
    {
        return true;
    }
    else if (update(state0, a, b, c))
    {
        return true;
    }
    else if (std::fabs(tc - a.t) < epsilon0<scalar_t>())
    {
        return  evaluate(state0, lsearch_step_t::interpolate(a0, a), c) ||
                update(state0, a, b, c);
    }
    else if (std::fabs(tc - b.t) < epsilon0<scalar_t>())
    {
        return  evaluate(state0, lsearch_step_t::interpolate(b0, b), c) ||
                update(state0, a, b, c);
    }
    else
    {
        return false;
    }
}

bool lsearchk_cgdescent_t::bracket(const solver_state_t& state0, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    assert(m_ro > 1);

    auto last_a = a;
    for (int i = 0; i < max_iterations(); ++ i)
    {
        if (!c.has_descent())
        {
            a = last_a;
            b = c;
            return false;
        }
        else if (!c.has_approx_armijo(state0, m_epsilon * m_sumC))
        {
            a = state0;
            b = c;
            return updateU(state0, a, b, c);
        }
        else
        {
            last_a = c;
            if (evaluate(state0, m_ro * c.t, c))
            {
                return true;
            }
        }
    }

    return false;
}

bool lsearchk_cgdescent_t::evaluate(const solver_state_t& state0, const scalar_t t, solver_state_t& c)
{
    // check overflow
    const auto ok = c.update(state0, t);
    log(state0, c);

    return (!ok) ? true : evaluate(state0, c);
}

bool lsearchk_cgdescent_t::evaluate(const solver_state_t& state0, const solver_state_t& state)
{
    if (!m_approx)
    {
        // check Armijo+Wolfe conditions
        if (state.has_armijo(state0, c1()) &&
            state.has_wolfe(state0, c2()))
        {
            // decide if to switch permanently to the approximate Wolfe conditions
            m_approx = std::fabs(state.f - state0.f) <= m_omega * m_sumC;
            return true;
        }
    }

    else
    {
        // check approximate Wolfe conditions
        if (state.has_approx_armijo(state0, m_epsilon * m_sumC) &&
            state.has_approx_wolfe(state0, c1(), c2()))
        {
            return true;
        }
    }

    return false;
}

bool lsearchk_cgdescent_t::get(const solver_state_t& state0, solver_state_t& state)
{
    // estimate an upper bound of the function value
    // (to be used for the approximate Wolfe condition)
    m_sumQ = 1 + m_sumQ * m_delta;
    m_sumC = m_sumC + (std::fabs(state0.f) - m_sumC) / m_sumQ;

    // evaluate the initial step length
    auto& c = state;
    if (evaluate(state0, state))
    {
        return true;
    }

    // bracket the initial step size
    lsearch_step_t a = state0, b = c;
    if (bracket(state0, a, b, c))
    {
        return true;
    }

    // iteratively update the search interval [a, b]
    for (int i = 0; i < max_iterations(); ++ i)
    {
        assert(a.t < b.t);
        assert(a.f <= state0.f + m_epsilon * m_sumC);
        assert(a.g < 0);
        assert(b.g >= 0);

        // secant interpolation
        const auto prev_width = b.t - a.t;
        if (secant2(state0, a, b, c))
        {
            return true;
        }

        // update search interval
        if (b.t - a.t > m_gamma * prev_width)
        {
            if (evaluate(state0, (a.t + b.t) / 2, c) ||
                update(state0, a, b, c))
            {
                return true;
            }
        }
    }

    return false;
}
