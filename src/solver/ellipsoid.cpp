#include <nano/solver/ellipsoid.h>

using namespace nano;

solver_ellipsoid_t::solver_ellipsoid_t()
{
    monotonic(false);

    register_parameter(parameter_t::make_scalar("solver::ellipsoid::R", 0.0, LT, 1e+1, LT, 1e+12));
}

solver_state_t solver_ellipsoid_t::minimize(const function_t& function_, const vector_t& x0) const
{
    const auto R = parameter("solver::ellipsoid::R").value<scalar_t>();
    const auto epsilon = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();

    auto function = make_function(function_, x0);
    auto state = solver_state_t{function, x0};

    auto y = state.x, gy = state.g;

    const auto n = static_cast<scalar_t>(function.size());

    matrix_t H = matrix_t::Identity(function.size(), function.size());
    H.array() *= R * R;

    for (int64_t i = 0; function.fcalls() < max_evals; ++ i)
    {
        const auto gHg = gy.dot(H * gy);

        y = y - (H * gy) / static_cast<scalar_t>(n + 1) / std::sqrt(gHg);
        H = (n * n) / (n * n - 1) * (H - 2.0 / (n + 1.0) * (H * gy * gy.transpose() * H) / gHg);

        const auto fy = function.vgrad(y, &gy);
        state.update_if_better(y, gy, fy);

        const auto iter_ok = std::isfinite(fy);
        const auto converged = std::sqrt(gHg) < epsilon;
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    return state;
}
