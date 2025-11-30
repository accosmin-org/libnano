#include <nano/critical.h>
#include <solver/interior.h>

using namespace nano;

solver_ipm_t::solver_ipm_t()
    : solver_t("ipm")
{
    register_parameter(parameter_t::make_scalar("solver::ipm::tau0", 0.0, LT, 0.9, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::gamma", 0.0, LT, 2.0, LE, 5.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::tiny_res", 0.0, LT, 1e-24, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::tiny_kkt", 0.0, LT, 1e-18, LE, 1.0));

    parameter("solver::max_evals") = 100;
}

rsolver_t solver_ipm_t::clone() const
{
    return std::make_unique<solver_ipm_t>(*this);
}

solver_state_t solver_ipm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    if (auto lconstraints = make_linear_constraints(function); !lconstraints)
    {
        raise("interior point solver can only solve linearly-constrained functions!");
    }

    // linear programs
    else if (const auto* const lprogram = dynamic_cast<const linear_program_t*>(&function); lprogram)
    {
        auto program = program_t{*lprogram, std::move(lconstraints.value()), x0};
        return do_minimize(program, logger);
    }

    // quadratic programs
    else if (const auto* const qprogram = dynamic_cast<const quadratic_program_t*>(&function); qprogram)
    {
        critical(is_convex(qprogram->Q()), "interior point solver can only solve convex quadratic programs!");

        auto program = program_t{*qprogram, std::move(lconstraints.value()), x0};
        return do_minimize(program, logger);
    }

    else
    {
        raise("interior point solver can only solve linear and convex quadratic programs!");
    }
}

solver_state_t solver_ipm_t::do_minimize(program_t& program, const logger_t& logger) const
{
    const auto tau0      = parameter("solver::ipm::tau0").value<scalar_t>();
    const auto gamma     = parameter("solver::ipm::gamma").value<scalar_t>();
    const auto tiny_res  = parameter("solver::ipm::tiny_res").value<scalar_t>();
    const auto tiny_kkt  = parameter("solver::ipm::tiny_kkt").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();

    const auto& function = program.function();

    auto bstate = solver_state_t{function, program.original_x()}; ///< best state (KKT optimality criterion-wise)
    auto cstate = bstate;                                         ///< current state

    // primal-dual interior-point solver...
    for (tensor_size_t iter = 1; function.fcalls() + function.gcalls() < max_evals; ++iter)
    {
        const auto tau = 1.0 - (1.0 - tau0) / std::pow(static_cast<scalar_t>(iter), gamma);

        // predictor-corrector update of primal-dual variables
        const auto stats = program.update(tau);

        logger.info("predictor: precision=", stats.m_predictor_stats.m_precision,
                    ",rcond=", stats.m_predictor_stats.m_rcond, ",valid=", stats.m_predictor_stats.m_valid ? 'y' : 'n',
                    ".\n");

        logger.info("corrector: precision=", stats.m_corrector_stats.m_precision,
                    ",rcond=", stats.m_corrector_stats.m_rcond, ",valid=", stats.m_corrector_stats.m_valid ? 'y' : 'n',
                    ".\n");

        logger.info("residual=", stats.m_residual, ",tau=", tau, ",sigma=", stats.m_sigma, ",alpha=", stats.m_alpha,
                    ".\n");

        if (!stats.m_valid)
        {
            logger.info("stopping as line-search step failed!\n");
            break;
        }

        // update current state
        cstate.update(program.original_x(), program.original_u(), program.original_v());
        done_kkt_optimality_test(cstate, cstate.valid(), logger);

        // update best state (if possible and an improvement)
        if (!cstate.valid())
        {
            logger.error("invalid current state after update!\n");
            break;
        }
        else if (iter == 1 || cstate.kkt_optimality_test() < bstate.kkt_optimality_test())
        {
            bstate = cstate;
        }

        // stop if no significant improvement
        if (stats.m_residual < tiny_res)
        {
            logger.info("stopping as residual too small!\n");
            break;
        }
        else if (bstate.kkt_optimality_test() < tiny_kkt)
        {
            logger.info("stopping as KKT optimality criteria too small!\n");
            break;
        }
    }

    // check convergence
    done_kkt_optimality_test(bstate, bstate.valid(), logger);

    return bstate;
}
