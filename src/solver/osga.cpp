#include <nano/solver/osga.h>

using namespace nano;

class proxy_t
{
public:

    proxy_t(const vector_t& z0) :
        m_z0(z0),
        m_Q0(0.5 * z0.lpNorm<2>() + std::numeric_limits<scalar_t>::epsilon())
    {
    }

    auto Q(const vector_t& z) const
    {
        return m_Q0 + 0.5 * (z - m_z0).dot(z - m_z0);
    }

    auto gQ(const vector_t& z) const
    {
        return z - m_z0;
    }

    auto E(scalar_t gamma, const vector_t& h) const
    {
        const auto beta = gamma + h.dot(m_z0);
        const auto sqrt = std::sqrt(beta * beta + 2.0 * m_Q0 * h.dot(h));

        if (beta <= 0.0)
        {
            return (-beta + sqrt) / (2.0 * m_Q0);
        }
        else
        {
            return h.dot(h) / (beta + sqrt);
        }
    }

    auto U(scalar_t gamma, const vector_t& h) const
    {
        return m_z0 - h / E(gamma, h);
    }

    auto UE(scalar_t gamma, const vector_t& h, scalar_t fx, scalar_t miu) const
    {
        return std::make_pair(U(gamma - fx, h), E(gamma - fx, h) - miu);
    }

private:

    const vector_t& m_z0;       ///<
    scalar_t        m_Q0{0.0};  ///<
};

solver_osga_t::solver_osga_t()
{
}

solver_state_t solver_osga_t::iterate(const solver_function_t& function, const lsearch_t&, const vector_t& x0) const
{
    return solver_state_t{function, x0};
}

solver_state_t solver_osga_t::minimize(const function_t& func, const vector_t& x0) const
{
    assert(func.size() == x0.size());

    auto function = solver_function_t{func};

    const auto delta = m_delta.get();
    const auto alpha_max = m_alpha_max.get();
    const auto kappa_prime = m_kappas.get1();
    const auto kappa = m_kappas.get2();

    const auto miu = func.strong_convexity() / 2.0;

    const auto proxy = proxy_t{x0};

    auto state = solver_state_t{function, x0};
    vector_t& xb = state.x;
    scalar_t& fb = state.f;
    vector_t& gb = state.g;

    vector_t x, x_prime, g, h, h_hat, u, u_hat, u_prime;
    scalar_t alpha = alpha_max, gamma, gamma_hat, eta, eta_hat, f, f_prime;

    h = gb - miu * proxy.gQ(xb);
    gamma = fb - miu * proxy.Q(xb) - h.dot(xb);
    std::tie(u, eta) = proxy.UE(gamma, h, fb, miu);

    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        x = xb + alpha * (u - xb);
        f = function.vgrad(x, &g);
        g = g - miu * proxy.gQ(x);

        h_hat = h + alpha * (g - h);
        gamma_hat = gamma + alpha * (f - miu * proxy.Q(x) - g.dot(x) - gamma);

        const auto& xb_prime = (f < fb) ? x : xb;
        const auto& fb_prime = (f < fb) ? f : fb;

        u_prime = proxy.U(gamma_hat - fb_prime, h_hat);
        x_prime = xb + alpha * (u_prime - xb);
        f_prime = function.vgrad(x_prime);

        const auto& xb_hat = (f_prime < fb_prime) ? x_prime : xb_prime;
        const auto& fb_hat = (f_prime < fb_prime) ? f_prime : fb_prime;

        std::tie(u_hat, eta_hat) = proxy.UE(gamma_hat, h_hat, fb_hat, miu);
        xb = xb_hat;

        // TODO: how to check convergence?!, implement various stopping criteria
        state.x = xb;
        state.f = fb;
        if (solver_t::done(function, state, true))
        {
            return state;
        }

        // algorithm 2.1, update (alpha, h, gamma, eta, u)
        const auto R = (eta - eta_hat) / (delta * alpha * eta);

        alpha = (R < 1.0) ?
            (alpha * std::exp(-kappa)) :
            std::min(alpha * std::exp(kappa_prime * (R - 1.0)), alpha_max);

        if (eta_hat < eta)
        {
            h = h_hat;
            u = u_hat;
            eta = eta_hat;
            gamma = gamma_hat;
        }
    }

    return state;
}
