#include <Eigen/Dense>
#include <nano/critical.h>
#include <nano/function/util.h>
#include <solver/interior.h>

using namespace nano;

namespace
{
auto make_init(const linear_constraints_t& lconstraints) -> std::tuple<vector_t, vector_t, vector_t>
{
    const auto& A = lconstraints.m_A;
    const auto& b = lconstraints.m_b;
    const auto& G = lconstraints.m_G;
    const auto& h = lconstraints.m_h;

    if (G.size() > 0)
    {
        // the starting point must be strictly feasible wrt inequality constraints
        if (const auto x00 = make_strictly_feasible(G, h); x00.has_value())
        {
            auto x0 = x00.value();
            auto u0 = vector_t{-1.0 / (G * x0 - h).array()};
            auto v0 = vector_t{vector_t::zero(b.size())};

            return std::make_tuple(std::move(x0), std::move(u0), std::move(v0));
        }
        else
        {
            return {};
        }
    }
    else
    {
        // the starting point is (one of the) solution to the linear equality constraints
        const auto Ab = Eigen::LDLT<eigen_matrix_t<scalar_t>>{(A.transpose() * A).matrix()};
        auto       x0 = vector_t{A.cols()};
        x0.vector()   = Ab.solve(A.transpose() * b);

        auto u0 = vector_t{};
        auto v0 = vector_t{vector_t::zero(b.size())};

        return std::make_tuple(std::move(x0), std::move(u0), std::move(v0));
    }
}

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

    // construct an appropriate initial point (if possible)
    else if (auto [x00, u0, v0] = make_init(*lconstraints); x00.size() == 0)
    {
        auto state = solver_state_t{function, x0};
        state.status(solver_status::unfeasible);
        done(state, state.valid(), state.status(), logger);
        return state;
    }

    // linear programs
    else if (const auto* const lprogram = dynamic_cast<const linear_program_t*>(&function); lprogram)
    {
        const auto miu = parameter("solver::ipm::miu").value<scalar_t>();

        auto program = program_t{*lprogram, std::move(lconstraints.value())};
        program.update(std::move(x00), std::move(u0), std::move(v0), miu);

        return do_minimize(program, logger);
    }

    // quadratic programs
    else if (const auto* const qprogram = dynamic_cast<const quadratic_program_t*>(&function); qprogram)
    {
        critical(is_convex(qprogram->Q()), "interior point solver can only solve convex quadratic programs!");

        const auto miu = parameter("solver::ipm::miu").value<scalar_t>();

        auto program = program_t{*qprogram, std::move(lconstraints.value())};
        program.update(std::move(x00), std::move(u0), std::move(v0), miu);

        return do_minimize(program, logger);
    }

    else
    {
        raise("interior point solver can only solve linear and convex quadratic programs!");
    }
}

solver_state_t solver_ipm_t::do_minimize(program_t& program, const logger_t& logger) const
{
    const auto s0                = parameter("solver::ipm::s0").value<scalar_t>();
    const auto miu               = parameter("solver::ipm::miu").value<scalar_t>();
    const auto beta              = parameter("solver::ipm::beta").value<scalar_t>();
    const auto alpha             = parameter("solver::ipm::alpha").value<scalar_t>();
    const auto epsilon0          = parameter("solver::ipm::epsilon0").value<scalar_t>();
    const auto lsearch_max_iters = parameter("solver::ipm::lsearch_max_iters").value<tensor_size_t>();
    const auto max_evals         = parameter("solver::max_evals").value<tensor_size_t>();

    const auto& G = program.G();
    const auto& h = program.h();
    const auto& function = program.function();

    auto state = solver_state_t{function, program.x()};

    // primal-dual interior-point solver...
    for (tensor_size_t iter = 1; function.fcalls() + function.gcalls() < max_evals; ++iter)
    {
        // solve primal-dual linear system of equations to get (dx, du, dv)
        const auto [valid, precision] = program.solve();

        if (!valid)
        {
            logger.error("linear system of equations cannot be solved, invalid state!\n");
            break;
        }

        logger.info("linear system of equations solved with accuracy=", precision, "\n");

        // line-search to reduce the KKT optimality criterion starting from the potentially different lengths
        // for the primal and dual steps: (x + sx * dx, u + su * du, v + su * dv)
        const auto s     = 1.0 - (1.0 - s0) / std::pow(static_cast<scalar_t>(iter), 2.0);
        const auto ustep = G.size() == 0 ? s : (s * make_umax(program.u(), program.du()));
        const auto xstep = G.size() == 0 ? s : (s * make_xmax(program.x(), program.dx(), G, h));
        const auto vstep = ustep;
        const auto kkt0  = program.kkt_test();

        auto lsearch_iter = 0;
        auto lsearch_step = 1.0;
        auto lsearch_kkt  = 0.0;

        for (; lsearch_iter < lsearch_max_iters; ++lsearch_iter)
        {
            lsearch_kkt = program.kkt_test(lsearch_step * xstep, lsearch_step * ustep, lsearch_step * vstep);
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
            break;
        }

        // update state
        program.update(lsearch_step * xstep, lsearch_step * ustep, lsearch_step * vstep, miu);
        state.update(program.x(), program.v(), program.u());
        done_kkt_optimality_test(state, state.valid(), logger);
    }

    // check convergence
    done_kkt_optimality_test(state, state.valid(), logger);

    return state;
}
