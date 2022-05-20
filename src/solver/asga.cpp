#include <nano/solver/asga.h>

using namespace nano;

static auto parameter_values(const estimator_t& estimator)
{
    return std::make_tuple(
        estimator.parameter("solver::epsilon").value<scalar_t>(),
        estimator.parameter("solver::max_evals").value<int64_t>(),
        estimator.parameter("solver::asga::L0").value<scalar_t>(),
        estimator.parameter("solver::asga::gamma1").value<scalar_t>(),
        estimator.parameter("solver::asga::gamma2").value<scalar_t>(),
        estimator.parameter("solver::asga::lsearch_max_iters").value<int64_t>());
}

static auto solve_sk1(scalar_t miu, scalar_t Sk, scalar_t Lk1)
{
    const auto r = 1.0 + Sk * miu;
    return (r + std::sqrt(r * r + 4.0 * Lk1 * Sk * r)) / (2.0 * Lk1);
}

static auto lsearch_done(
    const vector_t& y, scalar_t fy,
    const vector_t& x, scalar_t fx, const vector_t& gx,
    scalar_t Lk, scalar_t alphak, scalar_t epsilon)
{
    return fy <= fx + gx.dot(y - x) + 0.5 * Lk * (y - x).squaredNorm() + 0.5 * alphak * epsilon;
}

static auto converged(const function_t& function, const solver_state_t& state,
    const vector_t& xk, scalar_t fxk, const vector_t& xk1, scalar_t fxk1, scalar_t epsilon)
{
    return function.smooth() ?
        state.convergence_criterion() < epsilon :
        ((xk1 - xk).lpNorm<Eigen::Infinity>() < epsilon && std::fabs(fxk1 - fxk) < epsilon);
}

solver_asga_t::solver_asga_t()
{
    monotonic(false);

    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::asga::L0", 0.0, LT, 1.0, LT, fmax));
    register_parameter(parameter_t::make_scalar("solver::asga::gamma1", 1.0, LT, 4.0, LT, fmax));
    register_parameter(parameter_t::make_scalar("solver::asga::gamma2", 0.0, LT, 0.9, LT, 1.0));
    register_parameter(parameter_t::make_integer("solver::asga::lsearch_max_iters", 10, LE, 100, LE, 1000));
}

solver_asga2_t::solver_asga2_t() = default;

solver_state_t solver_asga2_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto [epsilon, max_evals, L0, gamma1, gamma2, lsearch_max_iters] = parameter_values(*this);

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};

    scalar_t Lk = L0;
    scalar_t Sk = 0.0;
    scalar_t fxk = std::numeric_limits<scalar_t>::max();

    vector_t xk = x0, xk1 = x0, gxk1{x0.size()};
    vector_t zk = x0, zk1 = x0;
    vector_t yk, gyk{x0.size()};
    vector_t sum_skgyk = vector_t::Zero(x0.size());

    const auto miu = function.strong_convexity();

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        auto iter_ok = false;
        auto Lk1 = Lk / gamma1, fxk1 = fxk, fyk = fxk, Sk1 = Sk, sk1 = 0.0;
        for (int64_t p = 0;
            p < lsearch_max_iters && !iter_ok && std::isfinite(Lk1) && std::isfinite(fxk1) && std::isfinite(fyk); ++ p)
        {
            Lk1 *= gamma1;
            sk1 = solve_sk1(miu, Sk, Lk1);
            Sk1 = Sk + sk1;

            const auto alphak = sk1 / Sk1;

            yk = alphak * zk + (1.0 - alphak) * xk;
            fyk = function.vgrad(yk, &gyk);
            state.update_if_better(yk, gyk, fyk);

            zk1 = (x0 + sum_skgyk + sk1 * (miu * yk - gyk)) / (1.0 + miu * Sk1);
            xk1 = alphak * zk1 + (1.0 - alphak) * xk;
            fxk1 = function.vgrad(xk1, &gxk1);
            state.update_if_better(xk1, gxk1, fxk1);

            iter_ok =
                std::isfinite(Lk1) && std::isfinite(fxk1) && std::isfinite(fyk) &&
                lsearch_done(xk1, fxk1, yk, fyk, gyk, Lk1, alphak, epsilon);
        }

        const auto converged = ::converged(function, state, xk, fxk, xk1, fxk1, epsilon);
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }

        xk = xk1;
        zk = zk1;
        Sk = Sk1;
        fxk = fxk1;
        Lk = gamma2 * Lk1;
        sum_skgyk += sk1 * (miu * yk - gyk);
    }

    return state;
}

solver_asga4_t::solver_asga4_t() = default;

solver_state_t solver_asga4_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto [epsilon, max_evals, L0, gamma1, gamma2, lsearch_max_iters] = parameter_values(*this);

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};

    scalar_t Lk = L0;
    scalar_t Sk = 0.0;
    scalar_t fyk = std::numeric_limits<scalar_t>::max();

    vector_t yk = x0, yk1, uk1, vk = x0, xk1;
    vector_t gxk1{x0.size()}, gyk1{x0.size()};
    vector_t sum_skgk = vector_t::Zero(x0.size());

    const auto miu = function.strong_convexity();

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        auto iter_ok = false;
        auto Lk1 = Lk / gamma1, fyk1 = fyk, fxk1 = fyk, Sk1 = Sk, sk1 = 0.0;
        for (int64_t p = 0;
            p < lsearch_max_iters && !iter_ok && std::isfinite(Lk1) && std::isfinite(fxk1) && std::isfinite(fyk1); ++ p)
        {
            Lk1 *= gamma1;
            sk1 = solve_sk1(miu, Sk, Lk1);
            Sk1 = Sk + sk1;

            const auto alphak = sk1 / Sk1;

            xk1 = alphak * vk + (1.0 - alphak) * yk;
            fxk1 = function.vgrad(xk1, &gxk1);
            state.update_if_better(xk1, gxk1, fxk1);

            uk1 = (vk + sk1 * (miu * xk1 - gxk1)) / (1.0 + miu * sk1);
            yk1 = alphak * uk1 + (1.0 - alphak) * yk;
            fyk1 = function.vgrad(yk1, &gyk1);
            state.update_if_better(yk1, gyk1, fyk1);

            iter_ok =
                std::isfinite(Lk1) && std::isfinite(fxk1) && std::isfinite(fyk1) &&
                lsearch_done(yk1, fyk1, xk1, fxk1, gxk1, Lk1, alphak, epsilon);
        }

        const auto converged = ::converged(function, state, yk, fyk, yk1, fyk1, epsilon);
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }

        yk = yk1;
        Sk = Sk1;
        fyk = fyk1;
        Lk = gamma2 * Lk1;

        sum_skgk += sk1 * (miu * xk1 - gxk1);
        vk = (x0 + sum_skgk) / (1.0 + miu * Sk);
    }

    return state;
}
