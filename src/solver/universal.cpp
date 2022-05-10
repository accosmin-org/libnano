#include <nano/solver/universal.h>

using namespace nano;

#include <iomanip>
#include <iostream>

static auto bregman_mapping(const vector_t& x, const vector_t& gx, scalar_t M)
{
    return x - 1.0 / M * gx;
}

static void register_parameters(estimator_t& estimator, const char* name)
{
    static constexpr auto max = std::numeric_limits<scalar_t>::max();

    const auto name_D = scat("solver::", name, "::D");
    const auto name_L0 = scat("solver::", name, "::L0");
    const auto name_ls_max_iters = scat("solver::", name, "::lsearch_max_iterations");

    estimator.register_parameter(parameter_t::make_float(name_D, 0.0, LT, 1.0, LT, max));
    estimator.register_parameter(parameter_t::make_float(name_L0, 0.0, LT, 1.0, LT, max));
    estimator.register_parameter(parameter_t::make_integer(name_ls_max_iters, 10, LE, 50, LE, 100));
}

static auto parameter_values(const estimator_t& estimator, const char* name)
{
    return std::make_tuple(
        estimator.parameter("solver::epsilon").value<scalar_t>(),
        estimator.parameter("solver::max_evals").value<int64_t>(),
        estimator.parameter(scat("solver::", name, "::D")).value<scalar_t>(),
        estimator.parameter(scat("solver::", name, "::L0")).value<scalar_t>(),
        estimator.parameter(scat("solver::", name, "::lsearch_max_iterations")).value<int64_t>());
}

solver_pgm_t::solver_pgm_t()
{
    monotonic(false);
    register_parameters(*this, "pgm");
}

solver_state_t solver_pgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto [epsilon, max_evals, D, L0, lsearch_max_iterations] = parameter_values(*this, "pgm");

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};
    auto bstate = state;        // best state so far
    auto& xp = state.d;         // line-search step
    const auto& f = state.f;
    const auto& x = state.x;
    const auto& g = state.g;

    scalar_t L = L0;
    scalar_t Sk = 0.0;
    vector_t Gk = vector_t::Zero(function.size());

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        // 1. line-search
        auto M = L, fp = f;
        auto iter_ok = false;
        auto converged = false;
        for (int64_t k = 0; k < lsearch_max_iterations && !iter_ok; ++ k, M *= 2.0)
        {
            xp = bregman_mapping(x, g, M);
            fp = function.vgrad(xp);
            iter_ok = std::isfinite(fp) && fp <= f + g.dot(xp - x) + 0.5 * M * (xp - x).dot(xp - x) + 0.5 * epsilon;
        }

        if (iter_ok)
        {
            Sk += 2.0 / M;
            Gk += state.g * 2.0 / M;

            // 2. update current and best state (if the line-search step doesn't fail)
            L = 0.5 * M;
            state.update(xp);
            if (state < bstate)
            {
                bstate.x = state.x;
                bstate.f = state.f;
                bstate.g = state.g;
            }

            // NB: There are two errors in the original paper regarding the convergence criterion:
            // * the definition of Sk
            // * the constraint eps(x0, y) <= D is applied with the wrong sign
            const auto beta = std::sqrt(Gk.dot(Gk) / (2.0 * D)) / Sk;
            converged = (Gk.dot(x0) / Sk + 2.0 * beta * D) < epsilon;

            const auto diff = Gk.dot(x0) / Sk + 2.0 * beta * D;

            // TODO: check convergence!!!
            //converged = (xp - x).lpNorm<Eigen::Infinity>() < epsilon && std::fabs(fp - f) < epsilon;


            std::cout << std::fixed << std::setprecision(12) << "i=" << i
                << ", f=" << state.f << "|" << bstate.f
                << ", g=" << state.g.lpNorm<Eigen::Infinity>() << "|" << bstate.g.lpNorm<Eigen::Infinity>()
                << ", L=" << L
                << ", calls=" << function.fcalls() << "|" << function.gcalls()
                << ", diff=" << diff
                << std::endl;
        }

        iter_ok = static_cast<bool>(state);
        if (solver_t::done(function, bstate, iter_ok, converged))
        {
            break;
        }
    }

    return bstate;
}

solver_dgm_t::solver_dgm_t()
{
    monotonic(false);
    register_parameters(*this, "dgm");
}

solver_state_t solver_dgm_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto [epsilon, max_evals, D, L0, lsearch_max_iterations] = parameter_values(*this, "dgm");

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};
    auto bstate = state;        // best state so far
    auto& xp = state.d;         // line-search step
    const auto& f = state.f;
    const auto& x = state.x;
    const auto& g = state.g;

    scalar_t L = L0;
    scalar_t Sk = 0.0;
    vector_t Gk = vector_t::Zero(function.size());

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        // 1. line-search
        auto M = L, fp = f;
        auto iter_ok = false;
        auto converged = false;
        for (int64_t k = 0; k < lsearch_max_iterations && !iter_ok; ++ k, M *= 2.0)
        {
            xp = bregman_mapping(x, g, M);
            fp = function.vgrad(xp);
            iter_ok = std::isfinite(fp) && fp <= f + g.dot(xp - x) + 0.5 * M * (xp - x).dot(xp - x) + 0.5 * epsilon;
        }

        if (iter_ok)
        {
            Sk += 2.0 / M;
            Gk += state.g * 2.0 / M;

            // 2. update current and best state (if the line-search step doesn't fail)
            L = 0.5 * M;
            state.update(xp);
            if (state < bstate)
            {
                bstate.x = state.x;
                bstate.f = state.f;
                bstate.g = state.g;
            }

            // NB: There are two errors in the original paper regarding the convergence criterion:
            // * the definition of Sk
            // * the constraint eps(x0, y) <= D is applied with the wrong sign
            const auto beta = std::sqrt(Gk.dot(Gk) / (2.0 * D)) / Sk;
            converged = (Gk.dot(x0) / Sk + 2.0 * beta * D) < epsilon;

            const auto diff = Gk.dot(x0) / Sk + 2.0 * beta * D;

            // TODO: check convergence!!!
            //converged = (xp - x).lpNorm<Eigen::Infinity>() < epsilon && std::fabs(fp - f) < epsilon;


            std::cout << std::fixed << std::setprecision(12) << "i=" << i
                << ", f=" << state.f << "|" << bstate.f
                << ", g=" << state.g.lpNorm<Eigen::Infinity>() << "|" << bstate.g.lpNorm<Eigen::Infinity>()
                << ", L=" << L
                << ", calls=" << function.fcalls() << "|" << function.gcalls()
                << ", diff=" << diff
                << std::endl;
        }

        iter_ok = static_cast<bool>(state);
        if (solver_t::done(function, bstate, iter_ok, converged))
        {
            break;
        }
    }

    return bstate;
}
