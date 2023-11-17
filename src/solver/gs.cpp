#include <nano/core/sampling.h>
#include <nano/program/solver.h>
#include <nano/solver/gs.h>

using namespace nano;

solver_gs_t::solver_gs_t()
    : solver_t("gs")
{
    type(solver_type::non_monotonic);
    parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1);

    register_parameter(parameter_t::make_scalar("solver::gs::gamma", 0, LT, 0.5, LT, 1));
    register_parameter(parameter_t::make_scalar("solver::gs::miu0", 0, LE, 0.1, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::gs::epsilon0", 0, LT, 0.1, LT, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::gs::theta_miu", 0, LT, 0.1, LE, 1));
    register_parameter(parameter_t::make_scalar("solver::gs::theta_epsilon", 0, LT, 0.1, LE, 1));
    register_parameter(parameter_t::make_integer("solver::gs::lsearch_max_iters", 0, LT, 50, LE, 100));
}

rsolver_t solver_gs_t::clone() const
{
    return std::make_unique<solver_gs_t>(*this);
}

solver_state_t solver_gs_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto max_evals         = parameter("solver::max_evals").value<tensor_size_t>();
    const auto epsilon           = parameter("solver::epsilon").value<scalar_t>();
    const auto gamma             = parameter("solver::gs::gamma").value<scalar_t>();
    const auto miu0              = parameter("solver::gs::miu0").value<scalar_t>();
    const auto epsilon0          = parameter("solver::gs::epsilon0").value<scalar_t>();
    const auto theta_miu         = parameter("solver::gs::theta_miu").value<scalar_t>();
    const auto theta_epsilon     = parameter("solver::gs::theta_epsilon").value<scalar_t>();
    const auto lsearch_max_iters = parameter("solver::gs::lsearch_max_iters").value<tensor_size_t>();

    const auto n = function.size();
    const auto m = n + 1;

    auto x        = vector_t{n};
    auto g        = vector_t{n};
    auto G        = matrix_t{m + 1, n};
    auto rng      = make_rng();
    auto miuk     = miu0;
    auto epsilonk = epsilon0;

    const auto positive = program::make_greater(m + 1, 0.0);
    const auto weighted = program::make_equality(vector_t::constant(m + 1, 1.0), 1.0);

    auto solver  = program::solver_t{};
    auto program = program::make_quadratic(matrix_t::zero(m + 1, m + 1), vector_t::zero(m + 1), positive, weighted);

    auto state  = solver_state_t{function, x0};
    auto cstate = state;

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // FIXME: should be more efficient for some functions to compute all gradients at once!
        // sample gradients within the given radius
        for (tensor_size_t i = 0; i < m; ++i)
        {
            sample_from_ball(state.x(), epsilonk, x, rng);
            assert((state.x() - x).lpNorm<2>() <= epsilonk);
            function.vgrad(x, map_tensor(G.row(i).data(), n));
        }
        G.row(m) = state.gx().transpose();

        // solve the quadratic problem to find the stabilized gradient
        program.m_Q = G * G.transpose();
        program.reduce();

        const auto solution = solver.solve(program);
        assert(solution.m_status == solver_status::converged);
        g = G.transpose() * solution.m_x.vector();

        // check convergence
        const auto iter_ok   = g.all_finite() && epsilonk > std::numeric_limits<scalar_t>::epsilon();
        const auto converged = state.gradient_test(g) < epsilon && epsilonk < epsilon;
        if (solver_t::done(state, iter_ok, converged))
        {
            break;
        }

        // too small gradient, reduce sampling radius (potentially convergence detected)
        else if (const auto gnorm2 = g.lpNorm<2>(); gnorm2 <= miuk)
        {
            miuk *= theta_miu;
            epsilonk *= theta_epsilon;
        }

        // line-search that handles functions that are not Lipschitz locally - see (4)
        else
        {
            x             = state.x();
            const auto fx = state.fx();

            auto              t    = 1.0;
            static const auto tmax = 1e+3;

            if (cstate.update(x - t * g); cstate.fx() < fx)
            {
                // doubling phase
                for (auto iters = 0; iters < lsearch_max_iters && t < tmax; ++iters)
                {
                    state = cstate;
                    if (t /= gamma, cstate.update(x - t * g); cstate.fx() >= fx)
                    {
                        break;
                    }
                }
            }
            else
            {
                // bisection phase
                for (auto iters = 0; iters < lsearch_max_iters; ++iters)
                {
                    if (t *= gamma, cstate.update(x - t * g); cstate.fx() < fx)
                    {
                        state = cstate;
                        break;
                    }
                }
            }

            /*if (state.fx() >= fx)
            {
                // NB: line-search failed, reduce the sampling radius - see (1)
                epsilonk *= theta_epsilon;
            }*/
        }
    }

    return state;
}
