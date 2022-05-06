#include <nano/solver/universal.h>

using namespace nano;

#include <iomanip>
#include <iostream>

static auto bregman_mapping(const vector_t& x, const vector_t& gx, scalar_t M)
{
    return x - 1.0 / M * gx;
}

solver_pgm_t::solver_pgm_t()
{
    monotonic(false);

    register_parameter(parameter_t::make_float("solver::pgm::L0", 0.0, LT, 1.0, LT, 1e+12));
    register_parameter(parameter_t::make_integer("solver::pgm::lsearch_max_iterations", 10, LE, 40, LE, 100));
}

solver_state_t solver_pgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto L0 = parameter("solver::pgm::L0").value<scalar_t>();
    const auto epsilon = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();
    const auto lsearch_max_iterations = parameter("solver::pgm::lsearch_max_iterations").value<int64_t>();

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};
    auto bstate = state;    // best state so far
    auto& xp = state.d;     // line-search step

    scalar_t L = L0;

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i, L *= 0.5)
    {
        // 1. line-search
        auto M = L;
        auto iter_ok = false;
        auto converged = false;
        for (int64_t k = 0; k < lsearch_max_iterations && !iter_ok; ++ k, M *= 2.0)
        {
            const auto& f = state.f;
            const auto& x = state.x;
            const auto& g = state.g;
            xp = bregman_mapping(x, g, M);

            const auto fp = function.vgrad(xp);
            iter_ok = std::isfinite(fp) && fp <= f + g.dot(xp - x) + 0.5 * M * (xp - x).dot(xp - x) + 0.5 * epsilon;
        }

        if (solver_t::done(function, bstate, iter_ok, converged))
        {
            break;
        }

        // 2. update current and best state (if the line-search step doesn't fail)
        state.update(xp);
        if (state < bstate)
        {
            bstate = state;
        }

        std::cout << std::fixed << std::setprecision(12) << "i=" << i
            << ", f=" << state.f << "|" << bstate.f
            << ", g=" << state.g.lpNorm<Eigen::Infinity>() << "|" << bstate.g.lpNorm<Eigen::Infinity>()
            << ", L=" << L
            << std::endl;

        // TODO: check convergence!!!
        iter_ok = static_cast<bool>(state);
        if (solver_t::done(function, bstate, iter_ok, converged))
        {
            break;
        }
    }

    return state;
}
