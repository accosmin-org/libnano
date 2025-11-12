#include <nano/solver.h>
#include <nano/tensor/stack.h>
#include <nano/function/util.h>
#include <nano/function/lambda.h>
#include <solver/interior/util.h>
#include <solver/interior/program.h>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

using namespace nano;

namespace
{
[[maybe_unused]] auto make_solver_LDLT(const matrix_t& lmat)
{
    return Eigen::LDLT<eigen_matrix_t<scalar_t>>{lmat.matrix()};
}

[[maybe_unused]] auto make_solver_BiCGSTAB(const matrix_t& lmat)
{
    return Eigen::BiCGSTAB<eigen_matrix_t<scalar_t>, Eigen::DiagonalPreconditioner<scalar_t>>{lmat.matrix()};
}

[[maybe_unused]] auto make_solver_CG(const matrix_t& lmat)
{
    return Eigen::ConjugateGradient<eigen_matrix_t<scalar_t>, Eigen::Lower | Eigen::Upper,
                                    Eigen::DiagonalPreconditioner<scalar_t>>{lmat.matrix()};
}

[[maybe_unused]] auto make_solver_MINRES(const matrix_t& lmat)
{
    return Eigen::MINRES<eigen_matrix_t<scalar_t>, Eigen::Lower | Eigen::Upper>{lmat.matrix()};
}

[[maybe_unused]] auto make_solver_GMRES(const matrix_t& lmat)
{
    return Eigen::GMRES<eigen_matrix_t<scalar_t>, Eigen::DiagonalPreconditioner<scalar_t>>{lmat.matrix()};
}

[[maybe_unused]] auto make_solver_DGMRES(const matrix_t& lmat)
{
    return Eigen::DGMRES<eigen_matrix_t<scalar_t>, Eigen::DiagonalPreconditioner<scalar_t>>{lmat.matrix()};
}

auto solve_kkt(const matrix_t& lmat, const vector_t& lvec, vector_t& lsol)
{
    const auto solver = make_solver_LDLT(lmat);

    // LBFGS (iterative)
    /*{
        const auto [D1, Ahat, D2] = ::ruiz_equilibration(m_lmat);

        const auto solver = solver_t::all().get("lbfgs");
        solver->parameter("solver::max_evals") = 10000;
        solver->parameter("solver::epsilon") = 1e-12;

        const auto lambda = [&](vector_cmap_t x, vector_map_t gx)
        {
            const auto b = (D1.array() * m_lvec.array()).matrix();

            if (gx.size() == x.size())
            {
                gx = Ahat * (Ahat * x - b);
            }
            return 0.5 * (Ahat * x - b).squaredNorm();
        };
        const auto function = make_function(m_lsol.size(), convexity::yes, smoothness::yes, 0.0, lambda);

        const auto state = solver->minimize(function, m_lvec, make_null_logger());
        m_lvec           = D2.array() * state.x().array();
        return m_lvec;
    }*/

    lsol.vector() = solver.solve(lvec.vector());

    // verify solution
    const auto valid    = lmat.all_finite() && lvec.all_finite() && lsol.all_finite();
    const auto delta    = (lmat * lsol - lvec).lpNorm<Eigen::Infinity>();
    const auto rcond    = solver.rcond();
    const auto positive = solver.isPositive();
    const auto negative = solver.isNegative();

    return program_t::solve_stats_t{delta, rcond, valid, positive, negative};
}
} // namespace

program_t::program_t(const linear_program_t& program, linear_constraints_t constraints, const vector_t& x0,
                     const scalar_t miu)
    : program_t(program, matrix_t{}, program.c(), std::move(constraints), x0, miu)
{
}

program_t::program_t(const quadratic_program_t& program, linear_constraints_t constraints, const vector_t& x0,
                     const scalar_t miu)
    : program_t(program, program.Q(), program.c(), std::move(constraints), x0, miu)
{
}

program_t::program_t(const function_t& function, matrix_t Q, vector_t c, linear_constraints_t constraints,
                     const vector_t& x0, const scalar_t miu)
    : m_function(function)
    , m_Q(std::move(Q))
    , m_c(std::move(c))
    , m_G(std::move(constraints.m_G))
    , m_h(std::move(constraints.m_h))
    , m_A(std::move(constraints.m_A))
    , m_b(std::move(constraints.m_b))
    , m_x(vector_t::zero(n() + m()))
    , m_u(vector_t::zero(m()))
    , m_v(vector_t::zero(p() + m()))
    , m_dx(vector_t::zero(m_x.size()))
    , m_du(vector_t::zero(m_u.size()))
    , m_dv(vector_t::zero(m_v.size()))
    , m_dQ(vector_t::constant(n(), 1.0))
    , m_dG(vector_t::constant(m(), 1.0))
    , m_dA(vector_t::constant(p(), 1.0))
    , m_rdual(n() + m())
    , m_rcent(m())
    , m_rprim(p() + m())
    , m_orig_x(n())
    , m_orig_u(m())
    , m_orig_v(p())
    , m_lmat(n() + p(), n() + p())
    , m_lvec(n() + p())
    , m_lsol(n() + p())
    , m_miu(miu)
{
    assert(m_Q.size() == 0 || m_Q.rows() == n());
    assert(m_Q.size() == 0 || m_Q.cols() == n());

    assert(m_c.size() == n());

    assert(m_A.rows() == p());
    assert(m_A.cols() == n());
    assert(m_b.size() == p());

    assert(m_G.rows() == m());
    assert(m_G.cols() == n());
    assert(m_h.size() == m());

    ::nano::modified_ruiz_equilibration(m_dQ, m_Q, m_c, m_dG, m_G, m_h, m_dA, m_A, m_b);

    m_x.segment(0, n())           = x0.vector();
    m_x.segment(n(), m()).array() = 1.0; // FIXME: have it parametrizable
    m_u.array()                   = 1.0; // FIXME: have it parametrizable

    update_original();
    update_residual();

    // FIXME: heuristic page 485 (numerical optimization book) to initialize (y, u)
    /*solve();

    m_x.segment(n(), m()).array() = (m_x.segment(n(), m()).array() + m_dx.segment(n(), m()).array()).abs().max(1.0);
    m_u.array()                   = (m_u.array() + m_du.array()).abs().max(1.0);

    update_original();
    update_residual();*/
}

program_t::solve_stats_t program_t::solve()
{
    const auto n = this->n();
    const auto m = this->m();
    const auto p = this->p();

    const auto y  = m_x.segment(n, m);
    const auto b1 = -m_rdual.segment(0, n);
    const auto b2 = -m_rdual.segment(n, m);
    const auto b3 = -m_rcent.segment(0, m);
    const auto b4 = -m_rprim.segment(0, p);
    const auto b5 = -m_rprim.segment(p, m);

    // |Q     0       0     A^T   G^T|   |dxn|   |-rdn|
    // |0     0      -I      0     I |   |dxm|   |-rdm|
    // |0  diag(u) diag(y)   0     0 | * |du | = |-rc |
    // |A     0       0      0     0 |   |dvp|   |-rpp|
    // |G     I       0      0     0 |   |dvm|   |-rpm|

    m_lmat.block(0, 0, n, n) = m_G.transpose() * (m_u.array() / y.array()).matrix().asDiagonal() * m_G;
    if (m_Q.size() > 0)
    {
        m_lmat.block(0, 0, n, n) += m_Q.matrix();
    }
    m_lmat.block(0, n, n, p) = m_A.transpose();
    m_lmat.block(n, 0, p, n) = m_A.matrix();
    m_lmat.block(n, n, p, p) = matrix_t::zero(p, p);

    const auto a7 = -m_u.array() * b5.array() + y.array() * b2.array() + b3.array();

    m_lvec.segment(0, n) = b1 - m_G.transpose() * (a7.array() / y.array()).matrix();
    m_lvec.segment(n, p) = b4;

    auto stats = solve_kkt(m_lmat, m_lvec, m_lsol);

    const auto dxn = m_lsol.segment(0, n);
    const auto dvp = m_lsol.segment(n, p);
    const auto dvm = a7.array() / y.array() + (m_u.array() / y.array()) * (m_G * dxn).array();

    m_dx.segment(0, n) = dxn;
    m_dx.segment(n, m) = b5 - m_G * dxn;
    m_du.segment(0, m) = dvm - b2.array();
    m_dv.segment(0, p) = dvp;
    m_dv.segment(p, m) = dvm;

    stats.m_valid = stats.m_valid && m_dx.all_finite() && m_du.all_finite() && m_dv.all_finite();
    return stats;
}

program_t::lsearch_stats_t program_t::lsearch(const scalar_t step0, const logger_t& logger)
{
    const auto n = this->n();
    const auto m = this->m();
    const auto p = this->p();

    const auto x = m_x.segment(0, n);
    const auto y = m_x.segment(n, m);
    const auto u = m_u.segment(0, m);
    const auto v = m_v.segment(0, p);
    const auto w = m_v.segment(p, m);

    const auto dx = m_dx.segment(0, n);
    const auto dy = m_dx.segment(n, m);
    const auto du = m_du.segment(0, m);
    const auto dv = m_dv.segment(0, p);
    const auto dw = m_dv.segment(p, m);

    const auto max_dstep = (m == 0) ? step0 : (step0 * make_umax(u, du));
    const auto max_pstep = (m == 0) ? step0 : (step0 * make_umax(y, dy));

    const auto residual0 = m_rdual.squaredNorm() + m_rprim.squaredNorm() + y.dot(u);

    // dual residual
    auto the_hx = matrix_t{2, 2};
    if (m_Q.size() == 0)
    {
        the_hx(0, 0) = 0.0;
        the_hx(1, 0) = 0.0;
    }
    else
    {
        the_hx(0, 0) = 2.0 * (m_Q * dx).dot(m_Q * dx);
        the_hx(1, 0) = 2.0 * (m_Q * dx).dot(m_A.transpose() * dv + m_G.transpose() * dw);
    }
    the_hx(0, 1) = the_hx(1, 0);
    the_hx(1, 1) = 2.0 * (dw - du).dot(dw - du) +
                   2.0 * (m_A.transpose() * dv + m_G.transpose() * dw).dot(m_A.transpose() * dv + m_G.transpose() * dw);

    // primal residual
    the_hx(0, 0) += 2.0 * (m_A * dx).dot(m_A * dx) + 2.0 * (m_G * dx + dy).dot(m_G * dx + dy);

    // centering residual
    if (m > 0)
    {
        the_hx(0, 1) += dy.dot(du);
        the_hx(1, 0) += du.dot(dy);
    }

    const auto make_residual = [&](const vector_cmap_t xx, vector_map_t gx, matrix_map_t hx)
    {
        const auto xstep = xx(0);
        const auto ystep = xx(0);
        const auto ustep = xx(1);
        const auto vstep = xx(1);
        const auto wstep = xx(1);

        if (xstep < 0.0 || xstep >= max_pstep ||
            ustep < 0.0 || ustep >= max_dstep)
        {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }

        auto fx = 0.0;

        // dual residual
        if (m_Q.size() == 0)
        {
            m_rdual.segment(0, n).matrix() = m_c.vector();
        }
        else
        {
            m_rdual.segment(0, n).matrix() = m_Q * (x + xstep * dx) + m_c;
        }
        m_rdual.segment(0, n) += m_A.transpose() * (v + vstep * dv);
        m_rdual.segment(0, n) += m_G.transpose() * (w + wstep * dw);
        m_rdual.segment(n, m) = (w + wstep * dw) - (u + ustep * du);

        fx += m_rdual.segment(0, n).squaredNorm();
        fx += m_rdual.segment(n, m).squaredNorm();

        if (gx.size() == xx.size())
        {
            if (m_Q.size() == 0)
            {
                gx(0) = 0.0;
            }
            else
            {
                gx(0) = 2.0 * m_rdual.segment(0, n).dot(m_Q * dx);
            }
            gx(1) = 2.0 * m_rdual.segment(0, n).dot(m_A.transpose() * dv + m_G.transpose() * dw) +
                    2.0 * m_rdual.segment(n, m).dot(dw - du);
        }

        // primal residual
        m_rprim.segment(0, p) = m_A * (x + xstep * dx) - m_b;
        m_rprim.segment(p, m) = m_G * (x + xstep * dx) + (y + ystep * dy) - m_h;

        fx += m_rprim.squaredNorm();

        if (gx.size() == xx.size())
        {
            gx(0) += 2.0 * m_rprim.segment(0, p).dot(m_A * dx) +
                     2.0 * m_rprim.segment(p, m).dot(m_G * dx + dy);
        }

        // centering residual
        if (m > 0)
        {
            fx += (y + ystep * dy).dot(u + ustep * du);

            if (gx.size() == xx.size())
            {
                gx(0) += dy.dot(u + ustep * du);
                gx(1) += du.dot(y + ystep * dy);
            }
        }

        if (hx.rows() == xx.size() && hx.cols() == xx.size())
        {
            hx = the_hx;
        }

        return fx;
    };

    const auto solver = solver_t::all().get("newton");
    solver->lsearchk("backtrack");

    const auto function = make_function(2, convexity::no, smoothness::yes, 0.0, make_residual);
    const auto x0       = make_vector<scalar_t>(0.9 * max_pstep, 0.9 * max_dstep);
    const auto state    = solver->minimize(function, x0, logger);

    const auto residual = state.fx();
    const auto xstep    = state.x()(0);
    const auto ystep    = state.x()(0);
    const auto ustep    = state.x()(1);
    const auto vstep    = state.x()(1);
    const auto wstep    = state.x()(1);

    m_x.segment(0, n) = x + xstep * dx;
    m_x.segment(n, m) = y + ystep * dy;
    m_u               = u + ustep * du;
    m_v.segment(0, p) = v + vstep * dv;
    m_v.segment(p, m) = w + wstep * dw;

    logger.info("residual=", residual0, "(pstep=", xstep, ",dstep=", ustep, ")->", residual, ".\n");

    update_original();
    update_residual();

    return lsearch_stats_t{xstep, ystep, ustep, vstep, wstep, residual, residual < residual0};

    /*const auto max_iters     = 30;
    const auto max_log_pstep = std::log10(pstep);
    const auto max_log_dstep = std::log10(dstep);
    const auto min_log_pstep = max_log_pstep - 4.0;
    const auto min_log_dstep = max_log_dstep - 4.0;

    auto best_residual   = std::numeric_limits<scalar_t>::max();
    auto best_xstep      = 0.0;
    auto best_ystep      = 0.0;
    auto best_ustep      = 0.0;
    auto best_vstep      = 0.0;
    auto best_wstep      = 0.0;
    auto best_log_pstep  = 0.5 * (min_log_pstep + max_log_pstep);
    auto best_log_dstep  = 0.5 * (min_log_dstep + max_log_dstep);
    auto delta_log_pstep = max_log_pstep - min_log_pstep;
    auto delta_log_dstep = max_log_dstep - min_log_dstep;

    const auto update_best = [&](const scalar_t log_pstep, const scalar_t log_dstep)
    {
        const auto xstep = std::min(std::pow(10.0, log_pstep), pstep);
        const auto ystep = std::min(std::pow(10.0, log_pstep), pstep);
        const auto ustep = std::min(std::pow(10.0, log_dstep), dstep);
        const auto vstep = std::min(std::pow(10.0, log_dstep), dstep);
        const auto wstep = std::min(std::pow(10.0, log_dstep), dstep);

        if (const auto residual = this->residual(xstep, ystep, ustep, vstep, wstep); residual < best_residual)
        {
            best_residual  = residual;
            best_xstep     = xstep;
            best_ystep     = ystep;
            best_ustep     = ustep;
            best_vstep     = vstep;
            best_wstep     = wstep;
            best_log_pstep = log_pstep;
            best_log_dstep = log_dstep;
        }
    };

    for (auto iters = 0; iters < max_iters; ++ iters)
    {
        const auto ref_log_pstep = best_log_pstep;
        const auto ref_log_dstep = best_log_dstep;

        update_best(ref_log_pstep - 0.5 * delta_log_pstep, ref_log_dstep - 0.5 * delta_log_dstep);
        update_best(ref_log_pstep - 0.5 * delta_log_pstep, ref_log_dstep + 0.0 * delta_log_dstep);
        update_best(ref_log_pstep - 0.5 * delta_log_pstep, ref_log_dstep + 0.5 * delta_log_dstep);
        update_best(ref_log_pstep + 0.0 * delta_log_pstep, ref_log_dstep - 0.5 * delta_log_dstep);
        update_best(ref_log_pstep + 0.0 * delta_log_pstep, ref_log_dstep + 0.0 * delta_log_dstep);
        update_best(ref_log_pstep + 0.0 * delta_log_pstep, ref_log_dstep + 0.5 * delta_log_dstep);
        update_best(ref_log_pstep + 0.5 * delta_log_pstep, ref_log_dstep - 0.5 * delta_log_dstep);
        update_best(ref_log_pstep + 0.5 * delta_log_pstep, ref_log_dstep + 0.0 * delta_log_dstep);
        update_best(ref_log_pstep + 0.5 * delta_log_pstep, ref_log_dstep + 0.5 * delta_log_dstep);

        logger.info("residual=", residual0, "(delta_log=", delta_log_pstep, ":", delta_log_dstep, ",iters=", iters, ")->", best_residual, ".\n");

        delta_log_pstep *= 0.5;
        delta_log_dstep *= 0.5;
    }

    m_x.segment(0, n) = x + best_xstep * dx;
    m_x.segment(n, m) = y + best_ystep * dy;
    m_u               = u + best_ustep * du;
    m_v.segment(0, p) = v + best_vstep * dv;
    m_v.segment(p, m) = w + best_wstep * dw;

    update_original();
    update_residual();

    return lsearch_stats_t{best_xstep, best_ystep, best_ustep, best_vstep, best_wstep, best_residual, best_residual < residual0};
    */
}

void program_t::update_original()
{
    const auto n = this->n();
    const auto m = this->m();
    const auto p = this->p();

    m_orig_x.array() = m_dQ.array() * m_x.segment(0, n).array();
    m_orig_u.array() = m_dG.array() * m_v.segment(p, m).array();
    m_orig_v.array() = m_dA.array() * m_v.segment(0, p).array();
}

void program_t::update_residual()
{
    const auto n = this->n();
    const auto m = this->m();
    const auto p = this->p();

    const auto x = m_x.segment(0, n);
    const auto y = m_x.segment(n, m);
    const auto u = m_u.segment(0, m);
    const auto v = m_v.segment(0, p);
    const auto w = m_v.segment(p, m);

    // dual residual
    if (m_Q.size() == 0)
    {
        m_rdual.segment(0, n).matrix() = m_c.vector();
    }
    else
    {
        m_rdual.segment(0, n).matrix() = m_Q * x + m_c;
    }
    m_rdual.segment(0, n) += m_A.transpose() * v;
    m_rdual.segment(0, n) += m_G.transpose() * w;

    m_rdual.segment(n, m) = w - u;

    // primal residual
    m_rprim.segment(0, p) = m_A * x - m_b;
    m_rprim.segment(p, m) = m_G * x + y - m_h;

    // centering residual
    if (m > 0)
    {
        assert(y.minCoeff() > 0);
        assert(u.minCoeff() > 0);

        m_rcent.array() = u.array() * y.array() - y.dot(u) / (m_miu * static_cast<scalar_t>(m));
    }
}
