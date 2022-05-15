#include <nano/solver/universal.h>

using namespace nano;

static auto parameter_values(const estimator_t& estimator)
{
    return std::make_tuple(
        estimator.parameter("solver::epsilon").value<scalar_t>(),
        estimator.parameter("solver::max_evals").value<int64_t>(),
        estimator.parameter("solver::universal::L0").value<scalar_t>(),
        estimator.parameter("solver::universal::lsearch_max_iters").value<int64_t>());
}

solver_universal_t::solver_universal_t()
{
    monotonic(false);

    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::universal::L0", 0.0, LT, 1.0, LT, fmax));
    register_parameter(parameter_t::make_integer("solver::universal::lsearch_max_iters", 10, LE, 50, LE, 100));
}

solver_pgm_t::solver_pgm_t() = default;

solver_state_t solver_pgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto [epsilon, max_evals, L0, lsearch_max_iterations] = parameter_values(*this);

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};

    auto L = L0;
    auto xk = state.x, xk1 = state.x;
    auto gxk = state.g, gxk1 = state.g;
    auto fxk = state.f, fxk1 = state.f;

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        // 1. line-search
        auto M = L;
        auto iter_ok = false;
        auto converged = false;
        for (int64_t k = 0; k < lsearch_max_iterations && !iter_ok && std::isfinite(fxk1); ++ k)
        {
            xk1 = xk - gxk / M;
            fxk1 = function.vgrad(xk1, &gxk1);
            iter_ok =
                std::isfinite(fxk1) &&
                fxk1 <= fxk + gxk.dot(xk1 - xk) + 0.5 * M * (xk1 - xk).dot(xk1 - xk) + 0.5 * epsilon;
            M *= 2.0;
        }

        if (iter_ok)
        {
            // 2. update current and best state (if the line-search step doesn't fail)
            L = 0.5 * M;
            xk = xk1;
            gxk = gxk1;
            fxk = fxk1;
            state.update_if_better(xk1, gxk1, fxk1);

            converged = function.smooth() ? (state.convergence_criterion() < epsilon) : false;
        }

        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    return state;
}

solver_dgm_t::solver_dgm_t() = default;

solver_state_t solver_dgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto [epsilon, max_evals, L0, lsearch_max_iterations] = parameter_values(*this);

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};

    auto L = L0;
    auto xk = state.x, xk1 = state.x, yk = state.x;
    auto gxk = state.g, gxk1 = state.g, gphi = x0;
    auto fxk1 = state.f;

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        // 1. line-search
        auto M = L;
        auto iter_ok = false;
        auto converged = false;
        for (int64_t k = 0; k < lsearch_max_iterations && !iter_ok && std::isfinite(fxk1); ++ k)
        {
            xk1 = gphi - gxk / M;
            fxk1 = function.vgrad(xk1, &gxk1);
            iter_ok =
                std::isfinite(fxk1) &&
                function.vgrad(yk = xk1 - gxk1 / M) <= fxk1 - 0.5 * gxk1.dot(gxk1) / M + 0.5 * epsilon;
            M *= 2.0;
        }

        if (iter_ok)
        {
            gphi -= gxk / M;

            // 2. update current and best state (if the line-search step doesn't fail)
            L = 0.5 * M;
            xk = xk1;
            gxk = gxk1;
            state.update_if_better(xk1, gxk1, fxk1);

            converged = function.smooth() ? (state.convergence_criterion() < epsilon) : false;
        }

        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    return state;
}

solver_fgm_t::solver_fgm_t() = default;

solver_state_t solver_fgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto [epsilon, max_evals, L0, lsearch_max_iterations] = parameter_values(*this);

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};

    auto L = L0, Ak = 0.0, ak1 = 0.0;
    auto vk = x0, yk = x0, yk1 = x0, xk1 = x0;
    auto gxk1 = state.g, gyk1 = state.g;
    auto fxk1 = state.f, fyk1 = state.f;

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        // 2. line-search
        auto M = L;
        auto iter_ok = false;
        auto converged = false;
        for (int64_t k = 0;
            k < lsearch_max_iterations && !iter_ok && std::isfinite(fxk1) && std::isfinite(fyk1);
            ++ k)
        {
            ak1 = (1.0 + std::sqrt(1.0 + 4.0 * M * Ak)) / (2.0 * M);
            const auto tau = ak1 / (Ak + ak1);

            xk1 = tau * vk + (1.0 - tau) * yk;
            fxk1 = function.vgrad(xk1, &gxk1);

            yk1 = tau * (vk - ak1 * gxk1) + (1.0 - tau) * yk;
            fyk1 = function.vgrad(yk1, &gyk1);

            iter_ok =
                std::isfinite(fxk1) &&
                std::isfinite(fyk1) &&
                fyk1 <= fxk1 + gxk1.dot(yk1 - xk1) + 0.5 * M * (yk1 - xk1).dot(yk1 - xk1) + 0.5 * epsilon * tau;
            M *= 2.0;
        }

        if (iter_ok)
        {
            // 3. update current and best state (if the line-search step doesn't fail)
            yk = yk1;
            Ak += ak1;
            L = 0.5 * M;
            vk -= ak1 * gxk1;

            state.update_if_better(yk1, gyk1, fyk1);

            converged = function.smooth() ? (state.convergence_criterion() < epsilon) : false;
        }

        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    return state;
}
