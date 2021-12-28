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

    vector_t y = x0, newy;
    vector_t sumg = vector_t::Zero(x0.size());

    auto& newx = state.d;
    auto& newxg = state.g;

    const auto L0 = 1.0;
    const auto lsearch_max_iterations = m_lsearch_max_iterations.get();

    scalar_t L = L0;
    scalar_t A = 0.0;
    scalar_t yf = std::numeric_limits<scalar_t>::max();

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

            newx = tau * v + (1.0 - tau) * y;
            const auto newxf = function.vgrad(newx, &newxg);
            newy = tau * (v - a * newxg) + (1.0 - tau) * y;

            const auto newyf = function.vgrad(newy);

            if (newyf <= newxf + newxg.dot(newy - newx) + 0.5 * M * (newy - newx).squaredNorm() + 0.5 * epsilon() * tau)
            {
                // TODO: implement proper stopping criterion, see paper
                iter_ok = true;
                converged =
                    (newy - y).lpNorm<Eigen::Infinity>() <= epsilon() * std::max(1.0, newy.lpNorm<Eigen::Infinity>()) &&
                    std::fabs(newyf - yf) <= epsilon() * std::max(1.0, std::fabs(newyf));

                if (newyf < state.f)
                {
                    state.x = newy;
                    state.f = newyf;
                }

                // 3. update state
                y = newy;
                yf = newyf;
                A = A + a;
                L = 0.5 * M;
                sumg += a * newxg;
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
