#include <nano/critical.h>
#include <solver/interior.h>

using namespace nano;

solver_ipm_t::solver_ipm_t()
    : solver_t("ipm")
{
    register_parameter(parameter_t::make_scalar("solver::ipm::tau0", 0.0, LT, 0.9, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::gamma", 0.0, LE, 2.0, LE, 5.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::accuracy_epsilon", 0.0, LT, 1e-7, LE, 1e-6));
    register_parameter(parameter_t::make_scalar("solver::ipm::residual_epsilon", 0.0, LT, 1e-18, LE, 1e-6));
    register_parameter(parameter_t::make_integer("solver::ipm::residual_patience", 0, LT, 5, LE, 100));

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
    const auto tau0             = parameter("solver::ipm::tau0").value<scalar_t>();
    const auto gamma            = parameter("solver::ipm::gamma").value<scalar_t>();
    const auto accuracy_epsilon = parameter("solver::ipm::accuracy_epsilon").value<scalar_t>();
    const auto residual_epsilon = parameter("solver::ipm::residual_epsilon").value<scalar_t>();
    const auto residual_patience = parameter("solver::ipm::residual_patience").value<tensor_size_t>();
    const auto max_evals        = parameter("solver::max_evals").value<tensor_size_t>();

    const auto& function = program.function();

    auto bstate = solver_state_t{function, program.original_x()}; ///< best state (KKT optimality criterion-wise)
    auto cstate = bstate;                                         ///< current state
    auto best_residual = std::numeric_limits<scalar_t>::max();

    // primal-dual interior-point solver...
    for (tensor_size_t iter = 1, best_iteration = 0; function.fcalls() + function.gcalls() < max_evals; ++iter)
    {
        const auto tau = 1.0 - (1.0 - tau0) / std::pow(static_cast<scalar_t>(iter), gamma);

        // predictor-corrector update of primal-dual variables
        const auto stats = program.update(tau, logger);

        logger.info("predictor: accuracy=", stats.m_predictor_stats.m_accuracy,
                    ",rcond=", stats.m_predictor_stats.m_rcond, ",valid=", stats.m_predictor_stats.m_valid ? 'y' : 'n',
                    ".\n");

        logger.info("corrector: accuracy=", stats.m_corrector_stats.m_accuracy,
                    ",rcond=", stats.m_corrector_stats.m_rcond, ",valid=", stats.m_corrector_stats.m_valid ? 'y' : 'n',
                    ".\n");

        logger.info("statistic: tau=", tau, ",sigma=", stats.m_sigma, ",alpha=", stats.m_alpha, ".\n");

        logger.info("statistic: res=", stats.m_primal_residual, "/", stats.m_dual_residual, ",gap=", stats.m_duality_gap, ".\n");

        if (!stats.m_valid)
        {
            logger.info("stopping as line-search step failed!\n");
            break;
        }
        if (std::max(stats.m_predictor_stats.m_accuracy, stats.m_corrector_stats.m_accuracy) > accuracy_epsilon)
        {
            logger.info("stopping as the KKT system wasn't solved precisely!\n");
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

        // check convergence
        const auto residual = std::max({stats.m_primal_residual, stats.m_dual_residual, stats.m_duality_gap});
        if (residual < residual_epsilon)
        {
            logger.info("stopping as the residuals and the duality gaps are too small!\n");
            break;
        }
        else if (residual < best_residual)
        {
            best_residual  = residual;
            best_iteration = 0;
        }
        else
        {
            ++best_iteration;
            if (best_iteration > residual_patience)
            {
                logger.info("stopping as the KKT optimality conditions haven't improved in the past iterations.\n");
                break;
            }
        }
    }

    // check convergence
    done_kkt_optimality_test(bstate, bstate.valid(), logger);

    return bstate;
}
