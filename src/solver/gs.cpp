#include <nano/core/sampling.h>
#include <nano/program/solver.h>
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

    const auto n = function.size();
    const auto m = n + 1;

    auto rng = make_rng();
    auto x   = vector_t{n};
    auto G   = matrix_t{m, n};

    const auto positive = program::make_greater(m, 0.0);
    const auto weighted = program::make_equality(vector_t::Constant(m, 1.0), 1.0);

    auto descent = vector_t{n};
    auto solver  = program::solver_t{};
    auto program =
        program::make_quadratic(matrix_t{matrix_t::Zero(m, m)}, vector_t{vector_t::Zero(m)}, positive, weighted);

    (void)(beta);
    (void)(gamma);
    (void)(miu0);
    (void)(epsilon0);
    (void)(theta_miu);
    (void)(theta_epsilon);

    // TODO: option to use the previous gradient as the starting point for QP
    // TODO: can it work with any line-search method?!
    assert(false);

    auto state = solver_state_t{function, x0};
    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // FIXME: should be more efficient for some functions to compute all gradients at once!
        // sample gradients within the given radius
        for (tensor_size_t i = 0; i < m; ++i)
        {
            sample_from_ball(state.x(), epsilon0, map_vector(x.data(), n), rng);
            function.vgrad(map_vector(x.data(), n), map_vector(G.row(i).data(), n));
        }

        // solve the quadratic problem to find the stabilized gradient
        program.m_Q         = G * G.transpose();
        const auto solution = solver.solve(program);
        assert(solution.m_status == solver_status::converged);

        descent = -G.transpose() * solution.m_x;

        // TODO: line-search

        // line-search
        const auto iter_ok = false; // lsearch.get(state, descent);
        if (solver_t::done(state, iter_ok, state.gradient_test() < epsilon))
        {
            break;
        }
    }

    return state;
} // LCOV_EXCL_LINE
