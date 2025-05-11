#include <Eigen/Dense>
#include <nano/critical.h>
#include <nano/function/cuts.h>
#include <nano/tensor/stack.h>
#include <solver/interior.h>
#include <solver/interior/util.h>

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
    if (auto lconstraints = make_linear_constraints(function); !lconstraints)
    {
        raise("interior point solver can only solve linearly-constrained functions!");
    }

    // linear programs
    else if (const auto* const lprogram = dynamic_cast<const linear_program_t*>(&function); lprogram)
    {
        auto program = program_t{*lprogram, std::move(lconstraints.value())};
        return do_minimize(program, x0, logger);
    }

    // quadratic programs
    else if (const auto* const qprogram = dynamic_cast<const quadratic_program_t*>(&function); qprogram)
    {
        critical(is_convex(qprogram->Q()), "interior point solver can only solve convex quadratic programs!");

        auto program = program_t{*qprogram, std::move(lconstraints.value())};
        return do_minimize(program, x0, logger);
    }

    else
    {
        raise("interior point solver can only solve linear and convex quadratic programs!");
    }
}

solver_state_t solver_ipm_t::do_minimize(program_t& program, const vector_t& x0, const logger_t& logger) const
{
    const auto& A   = program.A();
    const auto& b   = program.b();
    const auto& G   = program.G();
    const auto& h   = program.h();
    const auto  n   = program.n();
    const auto  m   = program.m();
    const auto  p   = program.p();
    const auto  miu = parameter("solver::ipm::miu").value<scalar_t>();

    if (G.size() > 0)
    {
        // phase 1: sum of infeasibilities program, see (2) p. 580
        auto fc = stack<scalar_t>(n + m, vector_t::zero(n), vector_t::constant(m, 1.0));

        auto fA = stack<scalar_t>(p, n + m, A, matrix_t::zero(p, m));
        auto fb = stack<scalar_t>(p, b);
        auto fG = stack<scalar_t>(m + m, n + m, G, -matrix_t::identity(m, m), matrix_t::zero(m, n),
                                  -matrix_t::identity(m, m));
        auto fh = stack<scalar_t>(m + m, h, vector_t::constant(m, 0.0));

        auto fobjective   = linear_program_t{"lp-phase1", std::move(fc)};
        critical(fA * fobjective.variable() == fb);
        critical(fG * fobjective.variable() <= fh);

        auto fconstraints = linear_constraints_t{std::move(fA), std::move(fb), std::move(fG), std::move(fh)};
        auto fprogram     = program_t{fobjective, std::move(fconstraints)};

        auto fx0 = stack<scalar_t>(n + m, x0, (G * x0 - h).array().max(0.0) + 1.0);
        auto fu0 = vector_t{-1.0 / (fprogram.G() * fx0 - fprogram.h()).array()};
        auto fv0 = vector_t{vector_t::zero(p)};

        assert((fprogram.G() * fx0 - fprogram.h()).maxCoeff() < 0.0);

        fprogram.update(std::move(fx0), std::move(fu0), std::move(fv0), miu);

        const auto state = do_minimize_phase2(fprogram, logger);
        if (state.valid() && state.kkt_optimality_test() < 1e-8 && std::fabs(state.fx()) < 1e-12)
        {
            auto ffx0 = vector_t{state.x().segment(0, n)};
            auto ffu0 = vector_t{-1.0 / (program.G() * ffx0 - program.h()).array()};
            auto ffv0 = vector_t{vector_t::zero(p)};

            assert((program.G() * ffx0 - program.h()).maxCoeff() < 0.0);

            program.update(std::move(ffx0), std::move(ffu0), std::move(ffv0), miu);
            return do_minimize_phase2(program, logger);
        }
    }
    else
    {
        // the starting point is (one of the) solution to the linear equality constraints
        const auto Ab  = Eigen::LDLT<eigen_matrix_t<scalar_t>>{(A.transpose() * A).matrix()};
        auto       fx0 = vector_t{n};
        fx0.vector()   = Ab.solve(A.transpose() * b);

        if ((A * fx0).isApprox(b.vector()))
        {
            auto fu0 = vector_t{};
            auto fv0 = vector_t{vector_t::zero(p)};

            program.update(std::move(fx0), std::move(fu0), std::move(fv0), miu);
            return do_minimize_phase2(program, logger);
        }
    }

    // no feasible point was found
    auto state = solver_state_t{program.function(), x0};
    state.status(solver_status::unfeasible);
    done(state, state.valid(), state.status(), logger);
    return state;
}

solver_state_t solver_ipm_t::do_minimize_phase2(program_t& program, const logger_t& logger) const
{
    const auto s0        = parameter("solver::ipm::s0").value<scalar_t>();
    const auto miu       = parameter("solver::ipm::miu").value<scalar_t>();
    const auto gamma     = parameter("solver::ipm::gamma").value<scalar_t>();
    const auto patience  = parameter("solver::ipm::patience").value<tensor_size_t>();
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();

    const auto& G        = program.G();
    const auto& h        = program.h();
    const auto& function = program.function();

    auto bstate = solver_state_t{function, program.x()}; ///< best state (KKT optimality criterion-wise)
    auto cstate = bstate;                                ///< current state
                                                         ///
    auto last_better_iter = tensor_size_t{0};

    // primal-dual interior-point solver...
    for (tensor_size_t iter = 1; function.fcalls() + function.gcalls() < max_evals; ++iter)
    {
        // solve primal-dual linear system of equations to get (dx, du, dv)
        if (const auto [valid, precision] = program.solve(); !valid)
        {
            logger.error("linear system of equations cannot be solved, invalid state!\n");
            break;
        }
        else
        {
            logger.info("linear system of equations solved with accuracy=", precision, ".\n");
        }

        // line-search to reduce the KKT optimality criterion starting from the potentially different lengths
        // for the primal and dual steps: (x + sx * dx, u + su * du, v + su * dv)
        const auto s     = 1.0 - (1.0 - s0) / std::pow(static_cast<scalar_t>(iter), gamma);
        const auto ustep = G.size() == 0 ? s : (s * make_umax(program.u(), program.du()));
        const auto xstep = G.size() == 0 ? s : (s * make_xmax(program.x(), program.dx(), G, h));
        const auto vstep = ustep;

        const auto curr_residual = program.residual();
        const auto next_residual = program.update(xstep, ustep, vstep, miu);

        logger.info("s=", s, "/", s0, ",step=(", xstep, ",", ustep, "),residual=", next_residual, "/", curr_residual,
                    ".\n");

        if (std::min({xstep, ustep, vstep}) < std::numeric_limits<scalar_t>::epsilon())
        {
            break;
        }

        // update current state
        program.update(xstep, ustep, vstep, miu, true);
        cstate.update(program.original_x(), program.original_v(), program.original_u());

        done_kkt_optimality_test(cstate, cstate.valid(), logger);

        // update best state (if possible)
        if (cstate.kkt_optimality_test() < bstate.kkt_optimality_test())
        {
            last_better_iter = 0;
            bstate           = cstate;
            continue;
        }

        if ((++last_better_iter) > patience)
        {
            break;
        }
    }

    // check convergence
    done_kkt_optimality_test(bstate, bstate.valid(), logger);

    return bstate;
}
