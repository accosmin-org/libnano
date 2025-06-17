#include <nano/critical.h>
#include <solver/interior.h>

using namespace nano;

solver_ipm_t::solver_ipm_t()
    : solver_t("ipm")
{
    register_parameter(parameter_t::make_scalar("solver::ipm::s0", 0.0, LT, 0.99, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::miu", 1.0, LT, 10.0, LE, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::ipm::gamma", 0.0, LT, 2.0, LE, 5.0));
    register_parameter(parameter_t::make_integer("solver::ipm::patience", 0, LT, 5, LE, 50));

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
        auto program = program_t{*lprogram, std::move(lconstraints.value()), x0, program_t::scale_type::ruiz, miu};
        return do_minimize(program, logger);
    }

    // quadratic programs
    else if (const auto* const qprogram = dynamic_cast<const quadratic_program_t*>(&function); qprogram)
    {
        critical(is_convex(qprogram->Q()), "interior point solver can only solve convex quadratic programs!");

        auto program = program_t{*qprogram, std::move(lconstraints.value()), x0, program_t::scale_type::ruiz, miu};
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
    const auto miu       = parameter("solver::ipm::miu").value<scalar_t>();
    const auto gamma     = parameter("solver::ipm::gamma").value<scalar_t>();
    const auto patience  = parameter("solver::ipm::patience").value<tensor_size_t>();
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();

    const auto& function = program.function();

    auto bstate = solver_state_t{function, program.original_x()}; ///< best state (KKT optimality criterion-wise)
    auto cstate = bstate;                                         ///< current state
                                                                  ///
    auto last_better_iter = tensor_size_t{0};

    // primal-dual interior-point solver...
    for (tensor_size_t iter = 1; function.fcalls() + function.gcalls() < max_evals; ++iter)
    {
        // solve primal-dual linear system of equations to get (dx, du, dv)
        const auto [precision, valid, rcond, positive, negative] = program.solve();
        logger.info("accuracy=", precision, ",rcond=", rcond, ",neg=", negative, ",pos=", positive, ",valid=", valid,
                    ".\n");
        if (!valid)
        {
            logger.error("linear system of equations cannot be solved, invalid state!\n");
            break;
        }

        // line-search to reduce the KKT optimality criterion starting from the potentially different lengths
        // for the primal and dual steps: (x + sx * dx, u + su * du, v + su * dv)
        const auto s     = 1.0 - (1.0 - s0) / std::pow(static_cast<scalar_t>(iter), gamma);
        const auto xstep = s;
        const auto vstep = s;
        const auto ustep = s * program.max_ustep();
        const auto ystep = s * program.max_ystep();

        const auto curr_residual = program.residual();
        const auto next_residual = program.update(xstep, ystep, ustep, vstep, miu);

        logger.info("step=", s, "/", s0, ",max_step=(", ystep, ",", ustep, "),residual=", next_residual, "/",
                    curr_residual, ".\n");

        if (std::min({ustep, ystep}) < std::numeric_limits<scalar_t>::epsilon())
        {
            break;
        }

        // update current state
        program.update(xstep, ystep, ustep, vstep, miu, true);
        cstate.update(program.original_x(), program.original_u(), program.original_v());

        done_kkt_optimality_test(cstate, cstate.valid(), logger);

        // update best state (if possible)
        if (!cstate.valid())
        {
            break;
        }
        else if (cstate.kkt_optimality_test() < bstate.kkt_optimality_test())
        {
            last_better_iter = 0;
            bstate           = cstate;
        }

        // stop if no significant improvement
        if ((++last_better_iter) > patience)
        {
            break;
        }
    }

    // check convergence
    done_kkt_optimality_test(bstate, bstate.valid(), logger);

    return bstate;
}
