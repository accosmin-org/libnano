#include <solver/bundle/bundle.h>
#include <solver/dsbm.h>

using namespace nano;

solver_dsbm_t::solver_dsbm_t()
    : solver_t("dsbm")
{
    register_parameter(parameter_t::make_scalar("solver::dsbm::ml", 0, LT, 0.2, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::dsbm::mf", 0, LT, 0.5, LT, 1.0));
    register_parameter(parameter_t::make_scalar_pair("solver::dsbm::tau_min_tau_one", 0, LT, 1e-6, LE, 1.0, LE, 1e+6));

    const auto prefix = string_t{"solver::dsbm"};
    bundle_t::config(*this, prefix);
}

rsolver_t solver_dsbm_t::clone() const
{
    return std::make_unique<solver_dsbm_t>(*this);
}

solver_state_t solver_dsbm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    warn_nonconvex(function, logger);
    warn_constrained(function, logger);

    const auto prefix    = string_t{"solver::dsbm"};
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();
    const auto ml        = parameter("solver::dsbm::ml").value<scalar_t>();
    const auto mf        = parameter("solver::dsbm::mf").value<scalar_t>();
    const auto [tau_min, tau1] = parameter("solver::dsbm::tau_min_tau_one").value_pair<scalar_t>();

    auto state  = solver_state_t{function, x0};
    auto bundle = bundle_t::make(state, *this, prefix);

    auto tau  = tau1;
    auto gxk1 = vector_t{function.size()};
    // auto nuL  = 1.0 + std::fabs(state.fx());                // NB: assumes no known lower bound
    // auto flow = std::numeric_limits<scalar_t>::quiet_NaN(); // NB: assumes no known lower bound

    auto flow = -100.0;
    auto nuL  = (1.0 - ml) * (bundle.fx() - flow);

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        const auto tol_delta = epsilon * (1.0 + std::fabs(bundle.fx()));
        const auto tol_error = epsilon * (1.0 + std::fabs(bundle.fx()));
        const auto tol_agrad = epsilon * 1e+2 * (1.0 + std::fabs(bundle.fx()));

        // FIXME: flow is not correct (-12.xxx) -> it should be <= -16!!!

        state.update_calls();
        logger.info(state, ",flow=", flow, ",tau=", tau, ",nuL=", nuL, ",delta=", (bundle.fx() - flow), ",bsize=", bundle.size(), ".\n");

        // first stopping criterion: optimality gap test
        if (const auto delta = bundle.fx() - flow; std::isfinite(delta) && delta < tol_delta)
        {
            const auto iter_ok   = state.valid();
            const auto converged = true;
            if (solver_t::done_specific_test(state, iter_ok, converged, logger))
            {
                break;
            }
        }

        // compute proximal point (with level constraint)
        const auto  level    = bundle.fx() - nuL;
        const auto& proximal = bundle.solve(tau, level, logger);

        if (proximal.m_status != solver_status::kkt_optimality_test)
        {
            // NB: no feasible solution, update level constraint!
            flow = level;
            nuL  = (1.0 - ml) * (bundle.fx() - flow);
            continue;
        }

        logger.info("level=", level, ",fxhat=", bundle.fhat(bundle.x()), ",fx1hat=", bundle.fhat(proximal.m_x), ".\n");

        // second stopping criterion: aggregate linearization error and gradient
        const auto& xk1   = proximal.m_x;
        const auto  miu   = 1.0 + proximal.m_lambda;
        const auto  nuT   = bundle.fx() - proximal.m_r;
        const auto  agrad = (bundle.x() - xk1) / (tau * miu);
        const auto  error = nuT - tau * miu * agrad.squaredNorm();

        // TODO: check consistency conditions (error >0, eq. 12-14)

        if (error <= tol_error && agrad.norm() <= tol_agrad)
        {
            const auto iter_ok   = state.valid();
            const auto converged = true;
            if (solver_t::done_specific_test(state, iter_ok, converged, logger))
            {
                break;
            }
        }

        // update state and bundle
        if (const auto fxk1 = function(xk1, gxk1); fxk1 <= bundle.fx() - mf * nuT)
        {
            logger.info("descent step...\n");

            // descent step: update center
            tau = tau * miu;
            nuL = std::min(nuL, (1.0 - ml) * (fxk1 - flow));

            state.update(xk1, gxk1, fxk1);
            bundle.moveto(xk1, gxk1, fxk1);
        }
        else
        {
            logger.info("null step...\n");

            // null step: update bundle model
            if (miu > 1.0)
            {
                nuL = ml * nuL;
            }
            bundle.append(xk1, gxk1, fxk1);
        }
    }

    state.update_calls();
    return state;
}
