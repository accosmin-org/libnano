#include <nano/critical.h>
#include <nano/function/util.h>
#include <solver/interior.h>
#include <solver/interior/state.h>

using namespace nano;

namespace
{
auto make_smax(const vector_t& u, const vector_t& du)
{
    assert(u.size() == du.size());

    auto smax = std::numeric_limits<scalar_t>::max();
    for (tensor_size_t i = 0, size = u.size(); i < size; ++i)
    {
        if (du(i) < 0.0)
        {
            smax = std::min(smax, -u(i) / du(i));
        }
    }

    return std::min(smax, 1.0);
}

bool converged(solver_state_t& state, const scalar_t epsilon)
{
    if (state.kkt_optimality_test() < epsilon)
    {
        state.status(solver_status::converged);
    }
    else if (state.feasibility_test() < epsilon)
    {
        // FIXME: this is an heuristic, to search for a theoretically sound method
        // to detect unboundness and unfeasibility!
        state.status(solver_status::unbounded);
    }
    else
    {
        // FIXME: this is an heuristic, to search for a theoretically sound method
        // to detect unboundness and unfeasibility!
        state.status(solver_status::unfeasible);
    }

    return state.status() == solver_status::converged;
}
} // namespace

solver_ipm_t::solver_ipm_t()
    : solver_t("ipm")
{
    register_parameter(parameter_t::make_scalar("solver::ipm::s0", 0.0, LT, 0.999, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::miu", 1.0, LT, 10.0, LE, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::ipm::alpha", 0.0, LT, 1e-2, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::beta", 0.0, LT, 0.9, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::epsilon0", 0.0, LE, 1e-16, LE, 1e-3));
    register_parameter(parameter_t::make_integer("solver::ipm::max_iters", 10, LE, 100, LE, 1000));
    register_parameter(parameter_t::make_integer("solver::ipm::max_lsearch_iters", 10, LE, 50, LE, 1000));
}

rsolver_t solver_ipm_t::clone() const
{
    return std::make_unique<solver_ipm_t>(*this);
}

solver_state_t solver_ipm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    if (const auto lconstraints = make_linear_constraints(function); !lconstraints)
    {
        raise("interior point solver can only solve linearly-constrained functions!");
    }
    else if (const auto* const lprogram = dynamic_cast<const linear_program_t*>(&function); lprogram)
    {
        return do_minimize(program_t{*lprogram, lconstraints.value()}, x0, logger);
    }
    else if (const auto* const qprogram = dynamic_cast<const quadratic_program_t*>(&function); qprogram)
    {
        critical(is_convex(qprogram->Q()), "interior point solver can only solver convex quadratic programs!");
        return do_minimize(program_t{*qprogram, lconstraints.value()}, x0, logger);
    }
    else
    {
        raise("interior point solver can only solve linear and quadratic programs!");
    }
}

solver_state_t solver_ipm_t::do_minimize(const program_t& program, const vector_t& x0, const logger_t& logger) const
{
    if (program.m() > 0)
    {
        const auto& G = program.m_G;
        const auto& h = program.m_h;

        // the starting point must be strictly feasible wrt inequality constraints
        if ((G * x0 - h).maxCoeff() >= 0.0)
        {
            if (const auto x00 = make_strictly_feasible(G, h); x00)
            {
                return do_minimize_with_inequality(program, x00.value(), logger);
            }
        }

        return do_minimize_with_inequality(program, x0, logger);
    }
    else
    {
        return do_minimize_without_inequality(program, x0, logger);
    }
}

solver_state_t solver_ipm_t::do_minimize_with_inequality(const program_t& program, const vector_t& x0,
                                                         const logger_t& logger) const
{
    const auto s0                = parameter("solver::ipm::s0").value<scalar_t>();
    const auto miu               = parameter("solver::ipm::miu").value<scalar_t>();
    const auto alpha             = parameter("solver::ipm::alpha").value<scalar_t>();
    const auto beta              = parameter("solver::ipm::beta").value<scalar_t>();
    const auto epsilon           = parameter("solver::epsilon").value<scalar_t>();
    const auto epsilon0          = parameter("solver::ipm::epsilon0").value<scalar_t>();
    const auto max_iters         = parameter("solver::ipm::max_iters").value<tensor_size_t>();
    const auto max_lsearch_iters = parameter("solver::ipm::max_lsearch_iters").value<tensor_size_t>();

    const auto& G = program.m_G;
    const auto& h = program.m_h;
    const auto  n = program.n();
    const auto  p = program.p();

    auto state = solver_state_t{*program.m_function, x0};

    // the starting point must be strictly feasible wrt inequality constraints
    if (const auto mGxh = (G * x0 - h).maxCoeff(); mGxh >= 0.0)
    {
        const auto iter_ok   = true;
        const auto converged = false;
        state.status(solver_status::unfeasible);
        solver_t::done(state, iter_ok, converged, logger);
        return state;
    }

    // the state of the primal-dual interior point iterations
    auto ipmst = state_t{x0, -1.0 / (G * x0 - h).array(), vector_t::zero(p)};

    // update residuals
    program.update(0.0, miu, ipmst);

    // primal-dual interior-point solver...
    for (ipmst.m_iters = 0; ipmst.m_iters < max_iters; ++ipmst.m_iters)
    {
        const auto prev_eta   = ipmst.m_eta;
        const auto prev_rdual = ipmst.m_rdual.lpNorm<2>();
        const auto prev_rprim = ipmst.m_rprim.lpNorm<2>();

        // solve primal-dual linear system of equations to get (dx, du, dv)
        const auto Gxh = G * ipmst.m_x - h;
        program.solve(G.transpose() * (ipmst.m_u.array() / Gxh.array()).matrix().asDiagonal() * G.matrix(),
                      ipmst.m_rdual + G.transpose() * (ipmst.m_rcent.array() / Gxh.array()).matrix(), ipmst.m_rprim);

        ipmst.m_dx            = program.m_lsol.segment(0, n);
        ipmst.m_dv            = program.m_lsol.segment(n, p);
        ipmst.m_du            = (ipmst.m_rcent.array() - ipmst.m_u.array() * (G * ipmst.m_dx).array()) / Gxh.array();
        ipmst.m_ldlt_rcond    = program.m_ldlt.rcond();
        ipmst.m_ldlt_positive = program.m_ldlt.isPositive();

        // stop if the linear system of equations is not stable
        if (!ipmst.isfinite())
        {
            const auto iter_ok   = state.valid();
            const auto converged = ::converged(state, epsilon);
            done(state, iter_ok, converged, logger);
            break;
        }

        // backtracking line-search: stage 1
        auto s    = s0 * make_smax(ipmst.m_u, ipmst.m_du);
        auto iter = tensor_size_t{0};
        for (iter = 0; iter < max_lsearch_iters; ++iter)
        {
            if ((G * (ipmst.m_x + s * ipmst.m_dx) - h).maxCoeff() < 0.0)
            {
                break;
            }
            else
            {
                s *= beta;
            }
        }
        if (iter == max_lsearch_iters)
        {
            const auto iter_ok   = state.valid();
            const auto converged = ::converged(state, epsilon);
            done(state, iter_ok, converged, logger);
            break;
        }

        // backtracking line-search: stage 2
        const auto r0 = ipmst.residual();
        for (iter = 0; iter < max_lsearch_iters; ++iter)
        {
            program.update(s, miu, ipmst);
            if (ipmst.residual() <= (1.0 - alpha * s) * r0)
            {
                break;
            }
            else
            {
                s *= beta;
            }
        }
        if (iter == max_lsearch_iters)
        {
            const auto iter_ok   = state.valid();
            const auto converged = ::converged(state, epsilon);
            done(state, iter_ok, converged, logger);
            break;
        }

        // update state
        ipmst.m_x += s * ipmst.m_dx;
        ipmst.m_u += s * ipmst.m_du;
        ipmst.m_v += s * ipmst.m_dv;
        state.update(ipmst.m_x, ipmst.m_v, ipmst.m_u);

        const auto curr_eta   = ipmst.m_eta;
        const auto curr_rdual = ipmst.m_rdual.lpNorm<2>();
        const auto curr_rprim = ipmst.m_rprim.lpNorm<2>();

        logger.info("i=", ipmst.m_iters, ",eta=", ipmst.m_eta, ",rdual=", ipmst.m_rdual.lpNorm<Eigen::Infinity>(),
                    ",rcent=", ipmst.m_rcent.lpNorm<Eigen::Infinity>(),
                    ",rprim=", ipmst.m_rprim.lpNorm<Eigen::Infinity>(), ",rcond=", ipmst.m_ldlt_rcond,
                    ipmst.m_ldlt_positive ? "(+)" : "(-)", ".\n");

        // check stopping criteria
        if (!std::isfinite(curr_eta) || !std::isfinite(curr_rdual) || !std::isfinite(curr_rprim))
        {
            // numerical instabilities
            const auto iter_ok   = state.valid();
            const auto converged = false;
            done(state, iter_ok, converged, logger);
            break;
        }
        else if (std::max({prev_eta - curr_eta, prev_rdual - curr_rdual, prev_rprim - curr_rprim}) < epsilon0)
        {
            // very precise convergence detected, check global convergence criterion!
            const auto iter_ok   = state.valid();
            const auto converged = ::converged(state, epsilon);
            done(state, iter_ok, converged, logger);
            break;
        }
        else
        {
            // not converged, continue the iterations
            const auto iter_ok   = state.valid();
            const auto converged = false;
            done(state, iter_ok, converged, logger);
        }
    }

    return state;
}

solver_state_t solver_ipm_t::do_minimize_without_inequality(const program_t& program, const vector_t& x0,
                                                            const logger_t& logger) const
{
    const auto miu     = parameter("solver::ipm::miu").value<scalar_t>();
    const auto epsilon = parameter("solver::epsilon").value<scalar_t>();

    const auto& c = program.m_c;
    const auto& b = program.m_b;
    const auto  n = program.n();
    const auto  p = program.p();

    auto state = solver_state_t{*program.m_function, x0};
    auto ipmst = state_t{x0, vector_t{}, vector_t::zero(p)};

    done(state, state.valid(), false, logger);

    // NB: solve directly the KKT-based system of linear equations coupling (x, v)
    program.solve(matrix_t::zero(n, n), c, -b);
    ipmst.m_x   = program.m_lsol.segment(0, n);
    ipmst.m_v   = program.m_lsol.segment(n, p);
    ipmst.m_eta = 0.0;

    program.update(0.0, miu, ipmst);
    state.update(ipmst.m_x, ipmst.m_v, ipmst.m_u);

    done(state, state.valid(), ::converged(state, epsilon), logger);

    return state;
}
