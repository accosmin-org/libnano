#include <nano/core/numeric.h>
#include <nano/lsearchk/cgdescent.h>

using namespace nano;

namespace
{
    struct interval_t
    {
        interval_t(const solver_state_t& state0, solver_state_t& state)
            : state0(state0)
            , c(state)
            , a(state0)
            , b(state)
        {
        }

        bool has_wolfe(scalar_t c1, scalar_t c2) const { return c.has_armijo(state0, c1) && c.has_wolfe(state0, c2); }

        bool has_approx_wolfe(scalar_t c1, scalar_t c2, scalar_t epsilonk) const
        {
            return c.has_approx_armijo(state0, epsilonk) && c.has_approx_wolfe(state0, c1, c2);
        }

        const solver_state_t& state0; ///< original point
        solver_state_t&       c;      ///< tentative point
        lsearch_step_t        a, b;   ///< lower/upper bounds of the bracketing interval
    };

    enum class status
    {
        exit, ///< exit criterion generated (Wolfe + approximate Wolfe)
        fail, ///< search failed
        done  ///< search succeeded, apply next step
    };

} // namespace

static auto initial_params(const estimator_t& estimator)
{
    const auto [c1, c2]       = estimator.parameter("lsearchk::tolerance").value_pair<scalar_t>();
    const auto max_iterations = estimator.parameter("lsearchk::max_iterations").value<int64_t>();
    const auto theta          = estimator.parameter("lsearchk::cgdescent::theta").value<scalar_t>();
    const auto epsilon        = estimator.parameter("lsearchk::cgdescent::epsilon").value<scalar_t>();
    const auto ro             = estimator.parameter("lsearchk::cgdescent::ro").value<scalar_t>();
    const auto omega          = estimator.parameter("lsearchk::cgdescent::omega").value<scalar_t>();
    const auto delta          = estimator.parameter("lsearchk::cgdescent::delta").value<scalar_t>();
    const auto gamma          = estimator.parameter("lsearchk::cgdescent::gamma").value<scalar_t>();
    const auto criterion =
        estimator.parameter("lsearchk::cgdescent::criterion").value<lsearchk_cgdescent_t::criterion_type>();

    return std::make_tuple(c1, c2, max_iterations, theta, epsilon, ro, omega, delta, gamma, crit);
}

static auto assert_interval([[maybe_unused]] const solver_state_t& state0, [[maybe_unused]] const lsearch_step_t& a,
                            [[maybe_unused]] const lsearch_step_t& b, [[maybe_unused]] const scalar_t epsilon,
                            [[maybe_unused]] const scalar_t Ck) const
{
    assert(a.f <= state0.f + epsilon * Ck);
    assert(a.g < 0.0);
    assert(b.g >= 0.0);
}

lsearchk_cgdescent_t::lsearchk_cgdescent_t()
{
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::epsilon", 0, LT, 1e-6, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::theta", 0, LT, 0.5, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::gamma", 0, LT, 0.66, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::delta", 0, LT, 0.7, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::omega", 0, LT, 1e-3, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearchk::cgdescent::ro", 1, LT, 5.0, LT, 1e+6));
    register_parameter(parameter_t::make_enum("lsearchk::cgdescent::criterion", criterion_type::wolfe_approx_wolfe));
}

rlsearchk_t lsearchk_cgdescent_t::clone() const
{
    auto lsearchk      = std::make_unique<lsearchk_cgdescent_t>(*this);
    lsearchk->Qk       = 0;
    lsearchk->Ck       = 0;
    lsearchk->m_approx = false;
    return rlsearchk_t{lsearchk.release()};
}

scalar_t lsearchk_cgdescent_t::approx_armijo_epsilon() const
{
    return parameter("lsearchk::cgdescent::epsilon").value<scalar_t>() * Ck;
}

bool lsearchk_cgdescent_t::get(const solver_state_t& state0, solver_state_t& state)
{
    const auto [c1, c2, max_iterations, theta, epsilon, ro, omega, delta, gamma, criterion] = initial_params(*this);

    // estimate an upper bound of the function value
    // (to be used for the approximate Wolfe condition)
    Qk = 1 + Qk * delta;
    Ck = Ck + (std::fabs(state0.f) - Ck) / Qk;

    auto& c = state;                  ///< tentative point
    auto  a = lsearch_step_t{state0}; ///< lower bound of the bracketing interval
    auto  b = lsearch_step_t{state0}; ///< upper bound of the bracketing interval

    const auto evaluate = [&](const scalar_t t)
    {
        const bool ok = c.update(state0, t);
        log(state0, c);

        return ok && evaluate();
    };

    const auto evaluate = [&]()
    {
        switch (criterion)
        {
        case criterion_type::wolfe: return interval.has_wolfe(c1, c2);

        case criterion_type::approx_wolfe: return interval.has_approx_armijo(c1, c2, epsilonk);

        default:
            if (!m_approx)
            {
                if (interval.has_wolfe(c1, c2))
                {
                    // decide if to switch permanently to the approximate Wolfe conditions
                    m_approx = std::fabs(state.f - state0.f) <= omega * Ck;
                    return true;
                }
                return false;
            }
            else
            {
                return interval.has_approx_armijo(c1, c2, epsilonk);
            }
        }
    };

    const auto updateU = [&]()
    {
        for (int64_t i = 0; i < max_iterations && (b.t - a.t) > stpmin(); ++i)
        {
            c.move((1 - theta) * a.t + theta * b.t);
            log(state0, c);
            if (!c)
            {
                break;
            }

            if (!c.has_descent())
            {
                b = c;
                return status::done;
            }
            else if (c.has_approx_armijo(state0, epsilon * Ck))
            {
                a = c;
            }
            else
            {
                b = c;
            }
        }

        return status::fail;
    };

    const auto update = [&]()
    {
        if (!c || c.t <= a.t || c.t >= b.t)
        {
            return status::done;
        }
        else if (!c.has_descent())
        {
            b = c;
            return status::done;
        }
        else if (c.has_approx_armijo(state0, epsilon * Ck))
        {
            a = c;
            return status::done;
        }
        else
        {
            b = c;
            return updateU();
        }
    };

    const auto secant2 = [&]()
    {
        const auto a0 = a, b0 = b;
        const auto tc = lsearch_step_t::secant(a0, b0);

        if (evaluate(state0, tc, c))
        {
            return status::exit;
        }

        const auto update_status = update();
        if (update_status != status::done)
        {
            return update_status;
        }
        else if (std::fabs(tc - a.t) < epsilon0<scalar_t>())
        {
            return evaluate(lsearch_step_t::secant(a0, a), c) ? status::exit : update(state0, a, b, c);
        }
        else if (std::fabs(tc - b.t) < epsilon0<scalar_t>())
        {
            return evaluate(lsearch_step_t::secant(b0, b), c) ? status::exit : update(state0, a, b, c);
        }
        else
        {
            return status::done;
        }
    };

    const auto bracket = [&]()
    {
        auto last_a = a;
        for (int64_t i = 0; i < max_iterations; ++i)
        {
            if (!c.has_descent())
            {
                a = last_a;
                b = c;
                break;
            }
            else if (!c.has_approx_armijo(state0, epsilon * Ck))
            {
                a = state0;
                b = c;
                updateU();
                break;
            }
            else
            {
                last_a = c;
                c.update(state0, ro * c.t);
                log(state0, c);
                if (!c)
                {
                    break;
                }
            }
        }
    };

    // evaluate the initial step length
    if (evaluate(state0, state))
    {
        return true;
    }

    // bracket the initial step size
    switch (bracket(state0, a, b, c))
    {
    case status::exit: return true;
    case status::fail: return false;
    default: break;
    }

    // iteratively update the search interval [a, b]
    for (int i = 0; i < max_iterations; ++i)
    {
        // secant interpolation
        const auto prev_width = b.t - a.t;
        switch (secant2(state0, a, b, c))
        {
        case status::exit: return true;
        case status::fail: return false;
        default: break;
        }

        // update search interval
        if (b.t - a.t > gamma * prev_width)
        {
            if (evaluate(state0, (a.t + b.t) / 2, c))
            {
                return true;
            }

            switch (update(state0, a, b, c))
            {
            case status::exit: return true;
            case status::fail: return false;
            default: break;
            }
        }
    }

    return false;
}
