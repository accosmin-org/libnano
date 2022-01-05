#include <nano/solver/fgm.h>

using namespace nano;

static auto compute_a(scalar_t M, scalar_t A)
{
    return 0.5 * (1.0 / M + std::sqrt(1.0 / (M * M) + 4.0 * A / M));
}

solver_fgm_t::solver_fgm_t()
{
    monotonic(false);
}

solver_state_t solver_fgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    auto function = make_function(function_, x0);

    auto state = solver_state_t{function};
    state.f = std::numeric_limits<scalar_t>::max();

    vector_t yk = x0, yk1;
    vector_t sumg = vector_t::Zero(x0.size());

    auto& xk1 = state.d;
    auto& gxk1 = state.g;

    const auto L0 = 1.0;
    const auto epsilon = std::numeric_limits<scalar_t>::epsilon();
    const auto lsearch_max_iterations = m_lsearch_max_iterations.get();

    scalar_t L = L0;
    scalar_t A = 0.0;
    scalar_t fyk = std::numeric_limits<scalar_t>::max();

    for (int64_t i = 0; i < max_iterations(); ++ i)
    {
        // 1.
        const auto v = x0 - sumg;

        // 2. line-search
        auto iter_ok = false;
        auto converged = false;
        for (int64_t k = 0; k < lsearch_max_iterations; ++ k)
        {
            const auto M = static_cast<scalar_t>(1 << k) * L;
            const auto a = compute_a(M, A);

            const auto tau = a / (A + a);

            xk1 = tau * v + (1.0 - tau) * yk;
            const auto fxk1 = function.vgrad(xk1, &gxk1);

            yk1 = tau * (v - a * gxk1) + (1.0 - tau) * yk;
            const auto fyk1 = function.vgrad(yk1);

            if (fyk1 <= fxk1 + gxk1.dot(yk1 - xk1) + 0.5 * M * (yk1 - xk1).squaredNorm() + 0.5 * epsilon * tau)
            {
                iter_ok = true;
                converged = solver_t::converged(yk, fyk, yk1, fyk1, state);

                // 3. update state
                yk = yk1;
                fyk = fyk1;
                A = A + a;
                L = 0.5 * M;
                sumg += a * gxk1;
                break;
            }
        }

        // 4. check stopping condition
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    // NB: make sure the gradient is updated at the returned point.
    state.f = function.vgrad(state.x, &state.g);
    return state;
}
