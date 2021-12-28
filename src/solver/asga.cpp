#include <nano/solver/asga.h>

using namespace nano;

auto solve_sk1(scalar_t miu, scalar_t Sk, scalar_t Lk1)
{
    const auto r = 1.0 + Sk * miu;
    return (r + std::sqrt(r * r + 4.0 * Lk1 * Sk * r)) / (2.0 * Lk1);
}

auto update_state(scalar_t Lk, scalar_t Sk,
    const vector_t& xk, scalar_t fxk, const vector_t& xk1, scalar_t fxk1, scalar_t epsilon, solver_state_t& state)
{
    const auto dx = (xk1 - xk).lpNorm<Eigen::Infinity>();
    const auto df = std::fabs(fxk1 - fxk);
    const auto converged =
        !std::isfinite(Lk) || !std::isfinite(Sk) ||
        (
            dx <= epsilon * std::max(1.0, xk1.lpNorm<Eigen::Infinity>()) &&
            df <= epsilon * std::max(1.0, std::fabs(fxk1))
        );

    if (std::isfinite(fxk1) && fxk1 < state.f)
    {
        state.x = xk1;
        state.f = fxk1;
    }

    return converged;
}

solver_asga2_t::solver_asga2_t()
{
    monotonic(false);
}

solver_state_t solver_asga2_t::minimize(const function_t& function_, const vector_t& x0) const
{
    auto function = make_function(function_, x0);

    const auto L0 = 1.0;
    const auto miu = function.strong_convexity();
    const auto eps = std::numeric_limits<scalar_t>::epsilon();

    const auto gamma1 = m_gamma1.get();
    const auto gamma2 = m_gamma2.get();
    const auto lsearch_max_iterations = m_lsearch_max_iterations.get();

    auto state = solver_state_t{function};
    state.f = std::numeric_limits<scalar_t>::max();

    scalar_t Lk = L0;
    scalar_t Sk = 0.0;
    scalar_t fxk = std::numeric_limits<scalar_t>::max();

    vector_t xk = x0, xk1 = x0;
    vector_t zk = x0, zk1 = x0;
    vector_t yk, gyk{x0.size()};
    vector_t sum_skgyk = vector_t::Zero(x0.size());

    for (int64_t k = 0; k < max_iterations(); ++ k)
    {
        auto iter_ok = false;
        auto Lk1 = Lk / gamma1, fxk1 = fxk, Sk1 = Sk, sk1 = 0.0;
        for (int64_t p = 0; p < lsearch_max_iterations && !iter_ok; ++ p)
        {
            Lk1 *= gamma1;
            sk1 = solve_sk1(miu, Sk, Lk1);
            Sk1 = Sk + sk1;

            const auto alphak = sk1 / Sk1;

            yk = alphak * zk + (1.0 - alphak) * xk;
            const auto fyk = function.vgrad(yk, &gyk);

            zk1 = (x0 + sum_skgyk + sk1 * (miu * yk - gyk)) / (1.0 + miu * Sk1);
            xk1 = alphak * zk1 + (1.0 - alphak) * xk;
            fxk1 = function.vgrad(xk1);

            iter_ok = fxk1 <= fyk + gyk.dot(xk1 - yk) + 0.5 * Lk1 * (xk1 - yk).squaredNorm() + 0.5 * alphak * eps;
        }

        const auto converged = update_state(Lk1, Sk1, xk, fxk, xk1, fxk1, epsilon(), state);

        xk = xk1;
        zk = zk1;
        Sk = Sk1;
        Lk = gamma2 * Lk1;
        fxk = fxk1;
        sum_skgyk += sk1 * (miu * yk - gyk);

        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    // NB: make sure the gradient is updated at the returned point.
    state.f = function.vgrad(state.x, &state.g);
    return state;
}

solver_asga4_t::solver_asga4_t()
{
    monotonic(false);
}

solver_state_t solver_asga4_t::minimize(const function_t& function_, const vector_t& x0) const
{
    auto function = make_function(function_, x0);

    const auto L0 = 1.0;
    const auto miu = function.strong_convexity();
    const auto eps = std::numeric_limits<scalar_t>::epsilon();

    const auto gamma1 = m_gamma1.get();
    const auto gamma2 = m_gamma2.get();
    const auto lsearch_max_iterations = m_lsearch_max_iterations.get();

    auto state = solver_state_t{function};
    state.f = std::numeric_limits<scalar_t>::max();

    scalar_t Lk = L0;
    scalar_t Sk = 0.0;
    scalar_t fyk = std::numeric_limits<scalar_t>::max();

    vector_t yk = x0, yk1 = x0;
    vector_t uk1;
    vector_t vk = x0;
    vector_t xk1, gxk1{x0.size()};
    vector_t sum_skgxk = vector_t::Zero(x0.size());

    for (int64_t k = 0; k < max_iterations(); ++ k)
    {
        auto iter_ok = false;
        auto Lk1 = Lk / gamma1, fyk1 = fyk, Sk1 = Sk, sk1 = 0.0;
        for (int64_t p = 0; p < lsearch_max_iterations && !iter_ok; ++ p)
        {
            Lk1 *= gamma1;
            sk1 = solve_sk1(miu, Sk, Lk1);
            Sk1 = Sk + sk1;

            const auto alphak = sk1 / Sk1;

            xk1 = alphak * vk + (1.0 - alphak) * yk;
            const auto fxk1 = function.vgrad(xk1, &gxk1);

            uk1 = (vk + sk1 * (miu * xk1 - gxk1)) / (1.0 + miu * sk1);
            yk1 = alphak * uk1 + (1.0 - alphak) * yk;
            fyk1 = function.vgrad(yk1);

            iter_ok = fyk1 <= fxk1 + gxk1.dot(yk1 - xk1) + 0.5 * Lk1 * (yk1 - xk1).squaredNorm() + 0.5 * alphak * eps;
        }

        const auto converged = update_state(Lk1, Sk1, yk, fyk, yk1, fyk1, epsilon(), state);

        yk = yk1;
        Sk = Sk1;
        Lk = gamma2 * Lk1;
        fyk = fyk1;

        sum_skgxk += sk1 * (miu * xk1 - gxk1);
        vk = (x0 + sum_skgxk) / (1.0 + miu * Sk);

        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    // NB: make sure the gradient is updated at the returned point.
    state.f = function.vgrad(state.x, &state.g);
    return state;
}
