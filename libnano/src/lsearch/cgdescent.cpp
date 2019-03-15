#include "cgdescent.h"
#include <nano/numeric.h>

using namespace nano;

json_t lsearch_cgdescent_t::config() const
{
    json_t json;
    json["epsilon"] = strcat(m_epsilon0, "(0,inf)");
    json["theta"] = strcat(m_theta, "(0,1)");
    json["gamma"] = strcat(m_gamma, "(0,1)");
    json["delta"] = strcat(m_delta, "[0,1]");
    json["omega"] = strcat(m_omega, "[0,1]");
    json["ro"] = strcat(m_ro, "(1,inf)");
    return json;
}

void lsearch_cgdescent_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();
    const auto inf = 1 / eps;

    nano::from_json_range(json, "epsilon", m_epsilon0, eps, inf);
    nano::from_json_range(json, "theta", m_theta, eps, 1 - eps);
    nano::from_json_range(json, "gamma", m_gamma, eps, 1 - eps);
    nano::from_json_range(json, "delta", m_delta, 0, 1);
    nano::from_json_range(json, "omega", m_omega, 0, 1);
    nano::from_json_range(json, "ro", m_ro, 1 + eps, inf);
}

bool lsearch_cgdescent_t::updateU(const solver_state_t& state0, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    assert(0 < m_theta && m_theta < 1);

    for (int i = 0; i < max_iterations(); ++ i)
    {
        if (evaluate(state0, (1 - m_theta) * a.t + m_theta * b.t, a, b, c))
        {
            return true;
        }
        else if (!c.has_descent())
        {
            b = c;
            return false;
        }
        else if (c.has_approx_armijo(state0, m_epsilon))
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

bool lsearch_cgdescent_t::update(const solver_state_t& state0, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
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
    else if (c.has_approx_armijo(state0, m_epsilon))
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

bool lsearch_cgdescent_t::secant2(const solver_state_t& state0, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
{
    const auto a0 = a, b0 = b;
    const auto tc = interpolate(a0, b0);

    if (evaluate(state0, tc, a, b, c))
    {
        return true;
    }
    else if (update(state0, a, b, c))
    {
        return true;
    }
    else if (std::fabs(tc - a.t) < epsilon0<scalar_t>())
    {
        return  evaluate(state0, lsearch_step_t::secant(a0, a), a, b, c) ||
                update(state0, a, b, c);
    }
    else if (std::fabs(tc - b.t) < epsilon0<scalar_t>())
    {
        return  evaluate(state0, lsearch_step_t::secant(b0, b), a, b, c) ||
                update(state0, a, b, c);
    }
    else
    {
        return false;
    }
}

bool lsearch_cgdescent_t::bracket(const solver_state_t& state0, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c)
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
        else if (!c.has_approx_armijo(state0, m_epsilon))
        {
            a = state0;
            b = c;
            return updateU(state0, a, b, c);
        }
        else
        {
            last_a = c;
            if (evaluate(state0, m_ro * c.t, a, b, c))
            {
                return true;
            }
        }
    }

    return false;
}

bool lsearch_cgdescent_t::evaluate(const solver_state_t& state0, const scalar_t t, solver_state_t& c)
{
    // check overflow
    if (!c.update(state0, t))
    {
        log(c);
        return true;
    }
    log(c);

    // check Armijo+Wolfe conditions or the approximate versions
    const auto done =
        (!m_approx && c.has_armijo(state0, c1()) && c.has_wolfe(state0, c2())) ||
        (m_approx && c.has_approx_armijo(state0, m_epsilon) && c.has_approx_wolfe(state0, c1(), c2()));

    if (done && !m_approx)
    {
        // decide if to switch permanently to the approximate Wolfe conditions
        m_approx = std::fabs(c.f - state0.f) <= m_omega * m_sumC;
    }

    return done;
}

bool lsearch_cgdescent_t::evaluate(const solver_state_t& state0, const scalar_t t,
    const lsearch_step_t& a, const lsearch_step_t& b, solver_state_t& c)
{
    if (evaluate(state0, t, c))
    {
        return true;
    }

    // check if the search interval is too small
    if (std::fabs(b.t - a.t) < epsilon0<scalar_t>())
    {
        return true;
    }

    // go on on updating the search interval
    return false;
}

bool lsearch_cgdescent_t::get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state)
{
    // estimate an upper bound of the function value
    // (to be used for the approximate Wolfe condition)
    m_sumQ = 1 + m_sumQ * m_delta;
    m_sumC = m_sumC + (std::fabs(state0.f) - m_sumC) / m_sumQ;
    m_epsilon = m_epsilon0 * m_sumC;

    // evaluate the initial step length
    auto& c = state;
    if (evaluate(state0, t0, c))
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
        // secant interpolation
        const auto prev_width = b.t - a.t;
        if (secant2(state0, a, b, c))
        {
            return true;
        }

        // update search interval
        if (b.t - a.t > m_gamma * prev_width)
        {
            if (evaluate(state0, (a.t + b.t) / 2, a, b, c) ||
                update(state0, a, b, c))
            {
                return true;
            }
        }
    }

    return false;
}
