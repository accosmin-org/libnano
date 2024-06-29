#include <lsearchk/cgdescent.h>
#include <nano/core/numeric.h>

using namespace nano;

struct lsearchk_cgdescent_t::params_t
{
    scalar_t    m_c1;
    scalar_t    m_c2;
    mutable int m_max_iterations;
    scalar_t    m_theta;
    scalar_t    m_ro;
    scalar_t    m_gamma;
    scalar_t    m_epsilonk;
};

namespace
{
auto make_params(const configurable_t& configurable, const solver_state_t& state0) -> lsearchk_cgdescent_t::params_t
{
    return {std::get<0>(configurable.parameter("lsearchk::tolerance").value_pair<scalar_t>()),
            std::get<1>(configurable.parameter("lsearchk::tolerance").value_pair<scalar_t>()),
            configurable.parameter("lsearchk::max_iterations").value<int>(),
            configurable.parameter("lsearchk::cgdescent::theta").value<scalar_t>(),
            configurable.parameter("lsearchk::cgdescent::ro").value<scalar_t>(),
            configurable.parameter("lsearchk::cgdescent::gamma").value<scalar_t>(),
            configurable.parameter("lsearchk::cgdescent::epsilon").value<scalar_t>() * std::fabs(state0.fx())};
}
} // namespace

struct lsearchk_cgdescent_t::interval_t
{
    interval_t(const solver_state_t& state0_, const vector_t& descent_, const scalar_t step_size_,
               solver_state_t& state)
        : state0(state0_)
        , descent(descent_)
        , step_size(step_size_)
        , c(state)
        , a(state0, descent, 0.0)
        , b(state, descent, step_size)
    {
    }

    void updateA() { a = {c, descent, step_size}; }

    void updateB() { b = {c, descent, step_size}; }

    bool done(const scalar_t c1, const scalar_t c2, const scalar_t epsilonk, const bool bracketed = true) const
    {
        assert(a.t <= b.t);
        assert(a.g < 0.0);

        if ((bracketed && (a.f > state0.fx() + epsilonk || b.g < 0.0)) || !c.valid())
        {
            // bracketing failed or diverged
            return true;
        }
        else if (step_size < a.t || step_size > b.t)
        {
            // tentative point outside bracketing interval, keep trying
            return false;
        }
        else
        {
            return
                // (Armijo-)Wolfe conditions
                (c.has_armijo(state0, descent, step_size, c1) && c.has_wolfe(state0, descent, c2)) ||

                // approximate (Armijo-)Wolfe conditions
                (c.has_approx_armijo(state0, epsilonk) && c.has_approx_wolfe(state0, descent, c1, c2));
        }
    }

    // attributes
    const solver_state_t& state0;    ///< original point
    const vector_t&       descent;   ///< descent direction
    scalar_t              step_size; ///< step size of the tentative point
    solver_state_t&       c;         ///< tentative point
    lsearch_step_t        a;         ///< lower bounds of the bracketing interval
    lsearch_step_t        b;         ///< upper bounds of the bracketing interval
};

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

void lsearchk_cgdescent_t::move(interval_t& interval, const scalar_t step_size_) const
{
    interval.step_size = step_size_;
    lsearchk_t::update(interval.c, interval.state0, interval.descent, interval.step_size);
}

void lsearchk_cgdescent_t::updateU(interval_t& interval, const params_t& params) const
{
    for (; params.m_max_iterations > 0 && (interval.b.t - interval.a.t) > stpmin(); --params.m_max_iterations)
    {
        move(interval, (1 - params.m_theta) * interval.a.t + params.m_theta * interval.b.t);
        if (!interval.c.valid())
        {
            return;
        }
        else if (!interval.c.has_descent(interval.descent))
        {
            interval.updateB();
            return;
        }
        else if (interval.c.has_approx_armijo(interval.state0, params.m_epsilonk))
        {
            interval.updateA();
        }
        else
        {
            interval.updateB();
        }
    }
}

void lsearchk_cgdescent_t::update(interval_t& interval, const params_t& params) const
{
    if (interval.step_size <= interval.a.t || interval.step_size >= interval.b.t)
    {
        return;
    }
    else if (!interval.c.has_descent(interval.descent))
    {
        interval.updateB();
    }
    else if (interval.c.has_approx_armijo(interval.state0, params.m_epsilonk))
    {
        interval.updateA();
    }
    else
    {
        interval.updateB();
        updateU(interval, params);
    }
}

void lsearchk_cgdescent_t::bracket(interval_t& interval, const params_t& params) const
{
    auto last_a = interval.a;
    for (; params.m_max_iterations > 0 && interval.c.valid(); --params.m_max_iterations)
    {
        if (!interval.c.has_descent(interval.descent))
        {
            interval.a = last_a;
            interval.updateB();
            return;
        }
        else if (!interval.c.has_approx_armijo(interval.state0, params.m_epsilonk))
        {
            interval.a = {interval.state0, interval.descent, 0.0};
            interval.updateB();
            updateU(interval, params);
            return;
        }
        else
        {
            last_a = {interval.c, interval.descent, interval.step_size};
            move(interval, params.m_ro * interval.step_size);
        }
    }
}

lsearchk_t::result_t lsearchk_cgdescent_t::do_get(const solver_state_t& state0, const vector_t& descent,
                                                  scalar_t step_size, solver_state_t& state) const
{
    assert(state0.has_descent(descent));

    const auto params = make_params(*this, state0);

    // current bracketing interval
    auto interval = interval_t{state0, descent, step_size, state};
    if (interval.done(params.m_c1, params.m_c2, params.m_epsilonk, false))
    {
        return {state.valid(), interval.step_size};
    }

    // bracket the initial step size
    bracket(interval, params);
    if (interval.done(params.m_c1, params.m_c2, params.m_epsilonk))
    {
        return {state.valid(), interval.step_size};
    }

    const auto move_update_and_check_done = [&](const auto t)
    {
        if (!std::isfinite(t))
        {
            // interpolation failed, go on with bisection?!
            return false;
        }

        move(interval, t);
        if (interval.done(params.m_c1, params.m_c2, params.m_epsilonk))
        {
            return true;
        }

        update(interval, params);
        return interval.done(params.m_c1, params.m_c2, params.m_epsilonk);
    };

    // iteratively update the search interval [a, b]
    for (int i = 0; i < params.m_max_iterations && (interval.b.t - interval.a.t) > stpmin(); ++i)
    {
        const auto& a = interval.a;
        const auto& b = interval.b;

        // secant interpolation
        const auto a0         = a;
        const auto b0         = b;
        const auto prev_width = b.t - a.t;

        if (const auto tc = lsearch_step_t::secant(a0, b0); move_update_and_check_done(tc))
        {
            return {state.valid(), interval.step_size};
        }
        else if (std::fabs(tc - a.t) < epsilon0<scalar_t>())
        {
            if (move_update_and_check_done(lsearch_step_t::secant(a0, a)))
            {
                return {state.valid(), interval.step_size};
            }
        }
        else if (std::fabs(tc - b.t) < epsilon0<scalar_t>())
        {
            if (move_update_and_check_done(lsearch_step_t::secant(b0, b)))
            {
                return {state.valid(), interval.step_size};
            }
        }

        // update search interval
        if (b.t - a.t > params.m_gamma * prev_width)
        {
            if (move_update_and_check_done((a.t + b.t) / 2))
            {
                return {state.valid(), interval.step_size};
            }
        }
    }

    return {false, interval.step_size};
}
