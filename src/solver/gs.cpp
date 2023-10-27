#include <nano/solver/gs.h>

using namespace nano;

solver_gs_t::solver_gs_t()
    : solver_t("gs")
{
    type(solver_type::line_search);
    parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1);

    register_parameter(parameter_t::make_scalar("solver::gs::beta", 0, LT, 1e-1, LT, 1));
    register_parameter(parameter_t::make_scalar("solver::gs::gamma", 0, LT, 9e-1, LT, 1));
    register_parameter(parameter_t::make_scalar("solver::gs::miu0", 0, LT, 1, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::gs::epsilon0", 0, LT, 1, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::gs::theta_miu", 0, LT, 0.9, LE, 1));
    register_parameter(parameter_t::make_scalar("solver::gs::theta_epsilon", 0, LT, 0.9, LE, 1));
}

rsolver_t solver_gs_t::clone() const
{
    return std::make_unique<solver_gs_t>(*this);
}

solver_state_t solver_gs_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto max_evals     = parameter("solver::max_evals").value<tensor_size_t>();
    const auto epsilon       = parameter("solver::epsilon").value<scalar_t>();
    const auto beta          = parameter("solver::gs::beta").value<scalar_t>();
    const auto gamma         = parameter("solver::gs::gamma").value<scalar_t>();
    const auto miu0          = parameter("solver::gs::miu0").value<scalar_t>();
    const auto epsilon0      = parameter("solver::gs::epsilon0").value<scalar_t>();
    const auto theta_miu     = parameter("solver::gs::theta_miu").value<scalar_t>();
    const auto theta_epsilon = parameter("solver::gs::theta_epsilon").value<scalar_t>();

    // TODO: can it work with any line-search method?!
    assert(false);

    auto state = solver_state_t{function, x0};
    if (solver_t::done(state, true, state.gradient_test() < epsilon))
    {
        return state;
    }

    auto lsearch = make_lsearch();
    auto descent = vector_t{function.size()};
    while (function.fcalls() + function.gcalls() < max_evals)
    {
        descent = -state.gx();

        // line-search
        const auto iter_ok = lsearch.get(state, descent);
        if (solver_t::done(state, iter_ok, state.gradient_test() < epsilon))
        {
            break;
        }
    }

    return state;
} // LCOV_EXCL_LINE
