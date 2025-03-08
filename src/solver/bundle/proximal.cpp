#include <solver/bundle/proximal.h>

using namespace nano;

namespace
{
scalar_t make_tau0(const solver_state_t& state, const scalar_t tau_min)
{
    // see (6)
    const auto tau0 = std::max(1.0, std::fabs(state.fx())) / (5.0 * state.gx().squaredNorm());
    return std::isfinite(tau0) ? std::max(tau0, tau_min) : tau_min;
}
} // namespace

proximal_t::proximal_t(const solver_state_t& state, const scalar_t tau_min, const scalar_t alpha,
                       const proximal_strategy strategy)
    : m_tau(make_tau0(state, tau_min))
    , m_tau_min(tau_min)
    , m_alpha(alpha)
    , m_strategy(strategy)
{
}

scalar_t proximal_t::tau() const
{
    assert(std::isfinite(m_tau));
    assert(m_tau > 0.0);
    return m_tau;
}

scalar_t proximal_t::miu() const
{
    return 1.0 / tau();
}

void proximal_t::update(const bundle_t& bundle, const csearch_t::point_t& point)
{
    // scale by the factor produced by the curve search
    m_tau *= point.m_t;

    // update descent step statistics
    const auto descent_step =
        (point.m_status == csearch_status::descent_step) || (point.m_status == csearch_status::cutting_plane_step);

    m_past_descent_steps = descent_step ? (m_past_descent_steps + 1) : 0;

    // compute auxiliary update named PBM-1/PBM-2 from (1)
    const auto tau_aux = [&]()
    {
        if (m_strategy == proximal_strategy::pbm1)
        {
            const auto tau_mul = (bundle.fx() - point.m_fy) / (bundle.fx() - point.m_fyhat);
            return 2.0 * m_tau * (1.0 + (std::isfinite(tau_mul) ? tau_mul : 0.0));
        }
        else
        {
            const auto delta_x = point.m_y - bundle.x();
            const auto delta_g = point.m_gy - bundle.gx();
            const auto tau_mul = delta_g.dot(delta_x) / delta_g.squaredNorm();
            return m_tau * (1.0 + (std::isfinite(tau_mul) ? tau_mul : 0.0));
        }
    }();

    // update decision from (1)
    if (!descent_step)
    {
        m_tau = std::min(m_tau, std::max({tau_aux, m_tau / m_alpha, m_tau_min}));
    }
    else if (m_past_descent_steps >= 5)
    {
        m_tau = std::min(m_alpha * tau_aux, 10.0 * m_tau);
    }
    else
    {
        m_tau = std::min(tau_aux, 10.0 * m_tau);
    }
}

void proximal_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::prox::tau_min"), 0.0, LT, 1e-5, LT, 1e+9));
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::prox::alpha"), 1.0, LT, 2.0, LT, 1e+3));
    c.register_parameter(parameter_t::make_enum(scat(prefix, "::prox::strategy"), proximal_strategy::pbm1));
}

proximal_t proximal_t::make(const solver_state_t& state, const configurable_t& c, const string_t& prefix)
{
    const auto tau_min  = c.parameter(scat(prefix, "::prox::tau_min")).value<scalar_t>();
    const auto alpha    = c.parameter(scat(prefix, "::prox::alpha")).value<scalar_t>();
    const auto strategy = c.parameter(scat(prefix, "::prox::strategy")).value<proximal_strategy>();

    return {state, tau_min, alpha, strategy};
}
