#include <nano/critical.h>
#include <nano/function/util.h>
#include <solver/interior.h>
#include <solver/interior/state.h>

using namespace nano;

namespace
{
template <class tvectoru, class tvectordu>
auto make_umax(const tvectoru& u, const tvectordu& du)
{
    assert(u.size() == du.size());

    auto step = std::numeric_limits<scalar_t>::max();
    for (tensor_size_t i = 0, size = u.size(); i < size; ++i)
    {
        if (du(i) < 0.0)
        {
            step = std::min(step, -u(i) / du(i));
        }
    }

    return std::min(step, 1.0);
}

auto make_xmax(const vector_t& x, const vector_t& dx, const matrix_t& G, const vector_t& h)
{
    assert(x.size() == dx.size());
    assert(x.size() == G.cols());
    assert(h.size() == G.rows());

    return make_umax(h - G * x, -G * dx);
}
} // namespace

solver_ipm_t::solver_ipm_t()
    : solver_t("ipm")
{
    register_parameter(parameter_t::make_scalar("solver::ipm::s0", 0.0, LT, 0.99, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::miu", 1.0, LT, 10.0, LE, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::ipm::beta", 0.0, LT, 0.7, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::alpha", 0.0, LT, 1e-5, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::ipm::epsilon0", 0.0, LT, 1e-24, LE, 1e-8));
    register_parameter(parameter_t::make_integer("solver::ipm::lsearch_max_iters", 10, LE, 100, LE, 1000));
}

rsolver_t solver_ipm_t::clone() const
{
    return std::make_unique<solver_ipm_t>(*this);
}

solver_state_t solver_ipm_t::do_minimize(const function_t& function, [[maybe_unused]] const vector_t& x0,
                                         const logger_t& logger) const
{
    if (auto lconstraints = make_linear_constraints(function); !lconstraints)
    {
        raise("interior point solver can only solve linearly-constrained functions!");
    }
    else if (const auto* const lprogram = dynamic_cast<const linear_program_t*>(&function); lprogram)
    {
        return do_minimize(program_t{*lprogram, std::move(lconstraints).value()}, logger);
    }
    else if (const auto* const qprogram = dynamic_cast<const quadratic_program_t*>(&function); qprogram)
    {
        critical(is_convex(qprogram->Q()), "interior point solver can only solve convex quadratic programs!");
        return do_minimize(program_t{*qprogram, std::move(lconstraints).value()}, logger);
    }
    else
    {
        raise("interior point solver can only solve linear and convex quadratic programs!");
    }
}

solver_state_t solver_ipm_t::do_minimize(const program_t& program, const logger_t& logger) const
{
    if (program.m() > 0)
    {
        return do_minimize_with_inequality(program, logger);
    }
    else
    {
        return do_minimize_without_inequality(program, logger);
    }
}

solver_state_t solver_ipm_t::do_minimize_with_inequality(const program_t& program, const logger_t& logger) const
{
    const auto miu       = parameter("solver::ipm::miu").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();

    const auto& G        = program.G();
    const auto& h        = program.h();
    const auto& function = program.function();
    const auto  n        = program.n();
    const auto  p        = program.p();

    // the starting point must be strictly feasible wrt inequality constraints
    const auto x00 = make_strictly_feasible(G, h);
    const auto x0  = x00.has_value() ? x00.value() : vector_t{vector_t::zero(n)};

    auto state = solver_state_t{function, x0};

    if (const auto mGxh = (G * x0 - h).maxCoeff(); mGxh >= 0.0)
    {
        state.status(solver_status::unfeasible);
        done(state, state.valid(), state.status(), logger);
        return state;
    }

    // the state of the primal-dual interior point iterations
    auto ipmst = state_t{x0, -1.0 / (G * x0 - h).array(), vector_t::zero(p)};

    // update residuals
    program.update(0.0, 0.0, 0.0, miu, ipmst);

    // primal-dual interior-point solver...
    for (tensor_size_t iter = 1; function.fcalls() + function.gcalls() < max_evals; ++iter)
    {
        // solve primal-dual linear system of equations to get (dx, du, dv)
        const auto  Gxh   = G * ipmst.m_x - h;
        const auto  hess  = G.transpose() * (ipmst.m_u.array() / Gxh.array()).matrix().asDiagonal() * G.matrix();
        const auto  rdual = ipmst.m_rdual + G.transpose() * (ipmst.m_rcent.array() / Gxh.array()).matrix();
        const auto& sol   = program.solve(hess, rdual, ipmst.m_rprim);

        ipmst.m_dx = sol.segment(0, n);
        ipmst.m_dv = sol.segment(n, p);
        ipmst.m_du = (ipmst.m_rcent.array() - ipmst.m_u.array() * (G * ipmst.m_dx).array()) / Gxh.array();

        // check solution
        if (!ipmst.valid())
        {
            logger.error("linear system of equations cannot be solved, invalid state!\n");
            break;
        }

        if (!program.valid())
        {
            logger.warn("linear system of equations not stable!\n");
        }

        // line-search for the primal and dual steps (x + sx * dx, u + su * du, v + su * dv)
        if (!lsearch(program, iter, state, ipmst, logger))
        {
            break;
        }
    }

    // check convergence
    done_kkt_optimality_test(state, state.valid(), logger);

    return state;
}

solver_state_t solver_ipm_t::do_minimize_without_inequality(const program_t& program, const logger_t& logger) const
{
    const auto miu       = parameter("solver::ipm::miu").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();

    const auto& A        = program.A();
    const auto& b        = program.b();
    const auto& function = program.function();
    const auto  n        = program.n();
    const auto  p        = program.p();

    // the starting point is (one of the) solution to the linear equality constraints
    const auto Ab = Eigen::LDLT<eigen_matrix_t<scalar_t>>{(A.transpose() * A).matrix()};
    auto       x0 = vector_t{n};
    x0.vector()   = Ab.solve(A.transpose() * b);

    auto state = solver_state_t{function, x0};

    // the state of the primal-dual interior point iterations
    auto ipmst = state_t{x0, vector_t{}, vector_t::zero(p)};

    // update residuals
    program.update(0.0, 0.0, 0.0, miu, ipmst);

    // primal-dual interior-point solver...
    for (tensor_size_t iter = 1; function.fcalls() + function.gcalls() < max_evals; ++iter)
    {
        // solve primal-dual linear system of equations to get (dx, du, dv)
        const auto& sol = program.solve(matrix_t::zero(n, n), ipmst.m_rdual, ipmst.m_rprim);

        ipmst.m_dx = sol.segment(0, n);
        ipmst.m_dv = sol.segment(n, p);

        // check solution
        if (!ipmst.valid())
        {
            logger.error("linear system of equations cannot be solved, invalid state!\n");
            break;
        }

        if (!program.valid())
        {
            logger.warn("linear system of equations not stable!\n");
        }

        // line-search for the primal and dual steps (x + sx * dx, u + su * du, v + su * dv)
        if (!lsearch(program, iter, state, ipmst, logger))
        {
            break;
        }
    }

    // check convergence
    done_kkt_optimality_test(state, state.valid(), logger);

    return state;
}

bool solver_ipm_t::lsearch(const program_t& program, const tensor_size_t iter, solver_state_t& state, state_t& ipmst,
                           const logger_t& logger) const
{
    const auto s0                = parameter("solver::ipm::s0").value<scalar_t>();
    const auto miu               = parameter("solver::ipm::miu").value<scalar_t>();
    const auto beta              = parameter("solver::ipm::beta").value<scalar_t>();
    const auto alpha             = parameter("solver::ipm::alpha").value<scalar_t>();
    const auto epsilon0          = parameter("solver::ipm::epsilon0").value<scalar_t>();
    const auto lsearch_max_iters = parameter("solver::ipm::lsearch_max_iters").value<tensor_size_t>();

    const auto& G = program.G();
    const auto& h = program.h();

    // line-search to reduce the KKT optimality criterion starting from the potentially different lengths
    // for the primal and dual steps: (x + sx * dx, u + su * du, v + su * dv)
    const auto s     = 1.0 - (1.0 - s0) / std::pow(static_cast<scalar_t>(iter), 2.0);
    const auto ustep = G.size() == 0 ? s : (s * make_umax(ipmst.m_u, ipmst.m_du));
    const auto xstep = G.size() == 0 ? s : (s * make_xmax(ipmst.m_x, ipmst.m_dx, G, h));
    const auto vstep = ustep;
    const auto kkt0  = program.kkt_test(ipmst);

    auto lsearch_iter = 0;
    auto lsearch_step = 1.0;
    auto lsearch_kkt  = 0.0;

    for (; lsearch_iter < lsearch_max_iters; ++lsearch_iter)
    {
        lsearch_kkt = program.kkt_test(lsearch_step * xstep, lsearch_step * ustep, lsearch_step * vstep, ipmst);
        if (lsearch_kkt <= (1.0 - alpha * lsearch_step) * kkt0)
        {
            break;
        }

        lsearch_step *= beta;
    }

    logger.info("s=", s, "/", s0, ",step=(", xstep, ",", ustep, "),lsearch=(iter=", lsearch_iter,
                ",step=", lsearch_step, ",kkt=", lsearch_kkt, "/", kkt0, ").\n");

    if (std::min({xstep, ustep, lsearch_step}) < std::numeric_limits<scalar_t>::epsilon() ||
        lsearch_iter >= lsearch_max_iters || std::fabs(lsearch_kkt - kkt0) < epsilon0)
    {
        return false;
    }

    // update state
    program.update(lsearch_step * xstep, lsearch_step * ustep, lsearch_step * vstep, miu, ipmst);
    state.update(ipmst.m_x, ipmst.m_v, ipmst.m_u);
    done_kkt_optimality_test(state, state.valid(), logger);
    return true;
}
