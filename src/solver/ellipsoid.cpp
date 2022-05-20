#include <nano/solver/ellipsoid.h>

using namespace nano;

solver_ellipsoid_t::solver_ellipsoid_t()
{
    monotonic(false);

    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::ellipsoid::R", 0.0, LT, 1e+1, LT, fmax));
}

solver_state_t solver_ellipsoid_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto R = parameter("solver::ellipsoid::R").value<scalar_t>();
    const auto epsilon = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};

    auto x = state.x, g = state.g;

    const auto n = static_cast<scalar_t>(function.size());

    matrix_t H = matrix_t::Identity(function.size(), function.size());
    H.array() *= R * R;

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        const auto gHg = g.dot(H * g);

        x.noalias() = x - (H * g) / static_cast<scalar_t>(n + 1) / std::sqrt(gHg);
        H = (n * n) / (n * n - 1) * (H - 2.0 / (n + 1.0) * (H * g * g.transpose() * H) / gHg);

        const auto f = function.vgrad(x, &g);
        state.update_if_better(x, g, f);

        const auto iter_ok = std::isfinite(f);
        const auto converged = std::sqrt(gHg / n) < epsilon;
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    return state;
}
