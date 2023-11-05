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

    auto x = vector_t{n};
    auto g = vector_t{n};
    auto G = matrix_t{m, n};
    auto rng = make_rng();
    auto miuk     = miu0;
    auto epsilonk = epsilon0;

    const auto positive = program::make_greater(m, 0.0);
    const auto weighted = program::make_equality(vector_t::constant(m, 1.0), 1.0);

    auto solver  = program::solver_t{};
    auto program = program::make_quadratic(matrix_t::zero(m, m), vector_t::zero(m), positive, weighted);

    // TODO: option to use the previous gradient as the starting point for QP
    // TODO: can it work with any line-search method?!
    // TODO: how to implement steps 8-10?!

    auto state = solver_state_t{function, x0};
    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // FIXME: should be more efficient for some functions to compute all gradients at once!
        // sample gradients within the given radius
        for (tensor_size_t i = 0; i < m; ++i)
        {
            sample_from_ball(state.x(), epsilonk, x, rng);
            function.vgrad(x, map_tensor(G.row(i).data(), n));
        }

        // solve the quadratic problem to find the stabilized gradient
        program.m_Q = G * G.transpose();
        program.reduce();

        const auto solution = solver.solve(program);
        assert(solution.m_status == solver_status::converged);
        g = G.transpose() * solution.m_x.vector();

        // check convergence
        const auto gnorm2    = g.lpNorm<2>();
        const auto iter_ok   = g.all_finite();
        const auto converged = gnorm2 <= epsilon && epsilonk <= epsilonk;
        if (solver_t::done(state, iter_ok, converged))
        {
            break;
        }

        // line-search
        if (gnorm2 <= miuk)
        {
            miuk *= theta_miu;
            epsilonk *= theta_epsilon;
        }
        else
        {
            const auto gsquared = g.squaredNorm();
            for (auto t = 1.0; t > std::numeric_limits<scalar_t>::epsilon(); t *= gamma)
            {
                x = state.x() - t * g;
                if (const auto fx = function.vgrad(x); fx < state.fx() - beta * t * gsquared)
                {
                    break;
                }
            }

            // FIXME: the functon value is computed multiple times at the same point during line-search!
            state.update(x);
        }
    }

    return state;
} // LCOV_EXCL_LINE
