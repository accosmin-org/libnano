#include <nano/critical.h>
#include <solver/interior.h>

using namespace nano;

solver_ipm_t::solver_ipm_t()
    : solver_t("ipm")
{
    register_parameter(parameter_t::make_scalar("solver::ipm::s0", 0.0, LT, 0.9, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::miu", 1.0, LT, 10.0, LE, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::ipm::gamma", 0.0, LT, 2.0, LE, 5.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::tiny", 0.0, LT, 1e-24, LE, 1.0));

    parameter("solver::max_evals") = 100;
}

rsolver_t solver_ipm_t::clone() const
{
    return std::make_unique<solver_ipm_t>(*this);
}

solver_state_t solver_ipm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    const auto miu = parameter("solver::ipm::miu").value<scalar_t>();

    if (auto lconstraints = make_linear_constraints(function); !lconstraints)
    {
        raise("interior point solver can only solve linearly-constrained functions!");
    }

    // linear programs
    else if (const auto* const lprogram = dynamic_cast<const linear_program_t*>(&function); lprogram)
    {
        auto program = program_t{*lprogram, std::move(lconstraints.value()), x0, miu};
        return do_minimize(program, logger);
    }

    // quadratic programs
    else if (const auto* const qprogram = dynamic_cast<const quadratic_program_t*>(&function); qprogram)
    {
        critical(is_convex(qprogram->Q()), "interior point solver can only solve convex quadratic programs!");

        auto program = program_t{*qprogram, std::move(lconstraints.value()), x0, miu};
        return do_minimize(program, logger);
    }

    else
    {
        raise("interior point solver can only solve linear and convex quadratic programs!");
    }
}

solver_state_t solver_ipm_t::do_minimize(program_t& program, const logger_t& logger) const
{
    const auto s0        = parameter("solver::ipm::s0").value<scalar_t>();
    const auto tiny      = parameter("solver::ipm::tiny").value<scalar_t>();
    const auto gamma     = parameter("solver::ipm::gamma").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();

    const auto& function = program.function();

    auto bstate = solver_state_t{function, program.original_x()}; ///< best state (KKT optimality criterion-wise)
    auto cstate = bstate;                                         ///< current state

    // primal-dual interior-point solver...
    for (tensor_size_t iter = 1; function.fcalls() + function.gcalls() < max_evals; ++iter)
    {
        // solve primal-dual linear system of equations to get (dx, dy, du, dv, dw)
        const auto sstats = program.solve();
        logger.info("accuracy=", sstats.m_precision, ",rcond=", sstats.m_rcond, ",pos=", sstats.m_positive ? 'y' : 'n',
                    ",valid=", sstats.m_valid ? 'y' : 'n', ".\n");
        if (!sstats.m_valid)
        {
            logger.error("linear system of equations cannot be solved, invalid state!\n");
            break;
        }

        // line-search to reduce the KKT optimality criterion starting from the potentially different lengths
        // for the primal and dual steps: (x + sx * dx, y + sy * dy, u + su * du, v + sv * dv)
        const auto s      = 1.0 - (1.0 - s0) / std::pow(static_cast<scalar_t>(iter), gamma);
        const auto lstats = program.lsearch(s, logger);
        logger.info("residual=", lstats.m_residual, ",success=", lstats.m_success ? 'y' : 'n', ".\n");
        if (!lstats.m_success)
        {
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
        if (lstats.m_residual < tiny)
        {
            break;
        }
    }

    // check convergence
    done_kkt_optimality_test(bstate, bstate.valid(), logger);

    return bstate;
}
