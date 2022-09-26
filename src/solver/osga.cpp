#include <nano/core/numeric.h>
#include <nano/solver/osga.h>

using namespace nano;

namespace
{
    class proxy_t
    {
    public:
        explicit proxy_t(const vector_t& z0, const scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon())
            : m_z0(z0)
            , m_Q0(0.5 * std::sqrt(z0.lpNorm<2>() + epsilon))
        {
        }

        auto Q(const vector_t& z) const { return m_Q0 + 0.5 * (z - m_z0).dot(z - m_z0); }

        auto gQ(const vector_t& z) const { return z - m_z0; }

        auto E(const scalar_t gamma, const vector_t& h) const
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

        auto U(const scalar_t gamma, const vector_t& h) const { return m_z0 - h / E(gamma, h); }

    private:
        const vector_t& m_z0;      ///<
        scalar_t        m_Q0{0.0}; ///<
    };
} // namespace

static auto converged(const vector_t& xk, const scalar_t fxk, const vector_t& xk1, const scalar_t fxk1,
                      const scalar_t epsilon)
{
    const auto dx = (xk1 - xk).lpNorm<Eigen::Infinity>();
    const auto df = std::fabs(fxk1 - fxk);

    return !std::isfinite(fxk) || !std::isfinite(fxk1) ||
           (dx > std::numeric_limits<scalar_t>::epsilon() &&
            dx < epsilon * std::max(1.0, xk1.lpNorm<Eigen::Infinity>()) &&
            df < epsilon * std::max(1.0, std::fabs(fxk1)));
}

solver_osga_t::solver_osga_t()
    : solver_t("osga")
{
    type(solver_type::non_monotonic);

    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::osga::lambda", 0, LT, 0.9, LT, 1));
    register_parameter(parameter_t::make_scalar("solver::osga::alpha_max", 0, LT, 0.7, LT, 1));
    register_parameter(parameter_t::make_scalar_pair("solver::osga::kappas", 0, LT, 0.5, LE, 0.5, LE, fmax));
}

rsolver_t solver_osga_t::clone() const
{
    return std::make_unique<solver_osga_t>(*this);
}

solver_state_t solver_osga_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto epsilon              = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals            = parameter("solver::max_evals").value<int>();
    const auto lambda               = parameter("solver::osga::lambda").value<scalar_t>();
    const auto alpha_max            = parameter("solver::osga::alpha_max").value<scalar_t>();
    const auto [kappa_prime, kappa] = parameter("solver::osga::kappas").value_pair<scalar_t>();

    const auto miu   = function.strong_convexity() / 2.0;
    const auto proxy = proxy_t{x0};

    auto  state = solver_state_t{function, x0}; // NB: keeps track of the best state
    auto& g     = state.g;                      // buffer to reuse

    // initialization
    vector_t h     = state.g - miu * proxy.gQ(state.x);
    scalar_t gamma = state.f - miu * proxy.Q(state.x) - h.dot(state.x);
    vector_t u     = proxy.U(gamma - state.f, h);
    scalar_t eta   = proxy.E(gamma - state.f, h) - miu;

    auto alpha = alpha_max;
    auto xb    = state.x;
    auto fb    = state.f;

    vector_t x, x_prime, h_hat, u_hat, u_prime;

    for (int i = 0; function.fcalls() < max_evals; ++i)
    {
        if (state.g.lpNorm<Eigen::Infinity>() < epsilon0<scalar_t>())
        {
            const auto converged = true;
            const auto iter_ok   = state.valid();
            if (solver_t::done(function, state, iter_ok, converged))
            {
                break;
            }
        }

        x            = xb + alpha * (u - xb);
        const auto f = function.vgrad(x, &g);
        g            = g - miu * proxy.gQ(x);

        h_hat                = h + alpha * (g - h);
        const auto gamma_hat = gamma + alpha * (f - miu * proxy.Q(x) - g.dot(x) - gamma);

        const auto& xb_prime = (f < fb) ? x : xb;
        const auto& fb_prime = (f < fb) ? f : fb;

        u_prime            = proxy.U(gamma_hat - fb_prime, h_hat);
        x_prime            = xb + alpha * (u_prime - xb);
        const auto f_prime = function.vgrad(x_prime);

        const auto& xb_hat = (f_prime < fb_prime) ? x_prime : xb_prime;
        const auto& fb_hat = (f_prime < fb_prime) ? f_prime : fb_prime;
        state.update_if_better(xb_hat, fb_hat);

        u_hat              = proxy.U(gamma_hat - fb_hat, h_hat);
        const auto eta_hat = proxy.E(gamma_hat - fb_hat, h_hat) - miu;

        // check convergence
        const auto converged = eta_hat <= epsilon0<scalar_t>() || ::converged(xb, fb, xb_hat, fb_hat, epsilon);
        const auto iter_ok   = state.valid();
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }

        // the algorithm to update the parameters (alpha, h, gamma, eta, u)
        const auto R = (eta - eta_hat) / (lambda * alpha * eta);

        alpha = (R < 1.0) ? (alpha * std::exp(-kappa)) : std::min(alpha * std::exp(kappa_prime * (R - 1.0)), alpha_max);
        if (eta_hat < eta)
        {
            h     = h_hat;
            u     = u_hat;
            eta   = eta_hat;
            gamma = gamma_hat;
        }

        xb = xb_hat;
        fb = fb_hat;
    }

    // NB: make sure the gradient is updated at the returned point.
    state.f      = function.vgrad(state.x, &state.g);
    state.fcalls = function.fcalls();
    state.gcalls = function.gcalls();
    return state;
}
