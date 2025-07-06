#include <Eigen/IterativeLinearSolvers>
#include <nano/function/util.h>
#include <nano/tensor/stack.h>
#include <solver/interior/program.h>
#include <solver/interior/util.h>
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

template <class tfunction>
auto newton_step(vector_t x0, const tfunction& function, const logger_t& logger)
{
    const auto beta              = 0.1;
    const auto alpha             = 1e-4;
    const auto gtol              = 1e-8;
    const auto max_iters         = 100;
    const auto max_lsearch_iters = 15;

    auto xx = std::move(x0);
    auto fx = 0.0;
    auto xp = vector_t{xx.size()};
    auto gx = vector_t{xx.size()};
    auto dx = vector_t{xx.size()};
    auto Hx = matrix_t{xx.size(), xx.size()};

    // NB: the Hessian is constant, so invert it once!
    function(xx, {}, Hx);
    const auto Hsolver = make_solver_LDLT(Hx.matrix());
    logger.info("Hsolver: dims=", Hx.dims(), ",pos=", Hsolver.isPositive(), ",neg=", Hsolver.isNegative(),
                ",rcond=", Hsolver.rcond(), "\n");

    // minimize sum of squares residuals (smooth convex function) wrt primal-dual steps
    //  - use gradient descent
    //  - w/  backtracking line-search
    for (auto iter = 0; iter < max_iters; ++iter)
    {
        fx          = function(xx, gx);
        dx.vector() = -gx.vector(); // Hsolver.solve(gx.vector());

        const auto gg = gx.lpNorm<Eigen::Infinity>() / (1.0 + std::fabs(fx));
        const auto dg = dx.dot(gx);

        logger.info("lsearch: i=", (iter + 1), "/", max_iters, ",residual=", fx, ",g=", gg, ",dg=", dg, "\n");

        assert(dg < epsilon0<scalar_t>());
        assert(std::isfinite(fx));
        assert(std::isfinite(gg));

        if (gg < gtol)
        {
            break;
        }

        auto lok   = false;
        auto lstep = 1.0;
        auto liter = 0;
        for (; liter < max_lsearch_iters; ++liter, lstep *= beta)
        {
            xp = xx + lstep * dx;

            const auto fxp = function(xp);
            logger.info("lsearch: i=", (iter + 1), "/", max_iters, ",li=", (liter + 1), "/", max_lsearch_iters,
                        ",lstep=", lstep, ",fx=", fxp, "/", fx, "\n");
            if (std::isfinite(fxp) && fxp <= fx + alpha * lstep * dg)
            {
                xx  = xp;
                fx  = fxp;
                lok = true;
                break;
            }
        }

        if (!lok)
        {
            logger.info("lsearch: line-search failed (lstep=", lstep, ",liter=", liter, ")!\n");
            break;
        }
    }

    return std::make_tuple(xx, fx);
}
} // namespace

program_t::program_t(const linear_program_t& program, linear_constraints_t constraints, const vector_t& x0,
                     const scale_type scale, const scalar_t miu)
    : program_t(program, matrix_t{}, program.c(), std::move(constraints), x0, scale, miu)
{
}

program_t::program_t(const quadratic_program_t& program, linear_constraints_t constraints, const vector_t& x0,
                     const scale_type scale, const scalar_t miu)
    : program_t(program, program.Q(), program.c(), std::move(constraints), x0, scale, miu)
{
}

program_t::program_t(const function_t& function, matrix_t Q, vector_t c, linear_constraints_t constraints,
                     const vector_t& x0, const scale_type scale, const scalar_t miu)
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

    switch (scale)
    {
    case scale_type::ruiz:
        ::nano::modified_ruiz_equilibration(m_dQ, m_Q, m_c, m_dG, m_G, m_h, m_dA, m_A, m_b);
        break;

    default:
        break;
    }

    m_x.segment(0, n())           = x0.vector();
    m_x.segment(n(), m()).array() = 1.0; // FIXME: have it parametrizable
    m_u.array()                   = 1.0; // FIXME: have it parametrizable

    update_original();
    update_residual();

    // TODO: heuristic page 485 to initialize (y, u)
    solve();

    m_x.segment(n(), m()).array() = (m_x.segment(n(), m()).array() + m_dx.segment(n(), m()).array()).abs().max(1.0);
    m_u.array()                   = (m_u.array() + m_du.array()).abs().max(1.0);

    update_original();
    update_residual();
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

program_t::lsearch_stats_t program_t::lsearch(const scalar_t s, const logger_t& logger)
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

    // allowed range of the primal step length
    const auto min_pstep = epsilon0<scalar_t>();
    const auto max_pstep = (m == 0) ? s : make_umax(y, dy, 1.0 - s);

    // allowed range of the dual step length (to make sure `u` and `y` are kept positive)
    const auto min_dstep = epsilon0<scalar_t>();
    const auto max_dstep = (m == 0) ? s : make_umax(u, du, 1.0 - s);

    // buffer constants
    const auto rdualQ0 = vector_t{m_Q * x + m_c};
    const auto rdualA0 = vector_t{m_A.transpose() * v};
    const auto rdualG0 = vector_t{m_G.transpose() * w};
    const auto rdualQx = vector_t{m_Q * dx};
    const auto rdualAv = vector_t{m_A.transpose() * dv};
    const auto rdualGw = vector_t{m_G.transpose() * dw};

    const auto rprimA0 = vector_t{m_A * x - m_b};
    const auto rprimAx = vector_t{m_A * dx};

    const auto rprimG0 = vector_t{m_G * x + y - m_h};
    const auto rprimGx = vector_t{m_G * dx};

    const auto rcent0  = u.dot(y);
    const auto rcentu  = du.dot(y);
    const auto rcenty  = dy.dot(u);
    const auto rcentuy = du.dot(dy);

    // residual as a function of the (primal, dual) step lengths
    const auto vgrad = [&](const vector_cmap_t xx, vector_map_t gx = {}, matrix_map_t Hx = {})
    {
        assert(xx.size() == 2);
        assert(gx.size() == 0 || xx.size() == gx.size());
        assert(Hx.size() == 0 || (xx.size() == Hx.rows() && xx.size() == Hx.cols()));

        const auto pstep = xx(0); // primal step length for updating (x, y)
        const auto dstep = xx(1); // dual step length for updating (u, v, w)

        // check if out-of bounds
        if ((pstep < min_pstep || pstep > max_pstep) || (dstep < min_dstep || dstep > max_dstep))
        {
            return std::numeric_limits<scalar_t>::quiet_NaN();
        }

        const auto rdual0 = rdualQ0 + rdualA0 + rdualG0 + pstep * rdualQx + dstep * (rdualAv + rdualGw);
        const auto rdual1 = w - u + dstep * (dw - du);
        const auto rprimA = rprimA0 + pstep * rprimAx;
        const auto rprimG = rprimG0 + pstep * (rprimGx + dy);
        const auto rcentr = rcent0 + dstep * rcentu + pstep * rcenty + dstep * pstep * rcentuy;

        if (gx.size() == xx.size())
        {
            gx(0) = rdual0.dot(rdualQx.vector()) + ///<
                    rprimA.dot(rprimAx.vector()) + ///<
                    rprimG.dot(rprimGx + dy) +     ///<
                    rcenty + dstep * rcentuy;      ///<

            gx(1) = rdual0.dot(rdualAv + rdualGw) + ///<
                    rdual1.dot(dw - du) +           ///<
                    rcentu + pstep * rcentuy;       ///<
        }

        if (Hx.rows() == xx.size() && Hx.cols() == xx.size())
        {
            Hx(0, 0) = rdualQx.dot(rdualQx) +            ///<
                       rprimAx.dot(rprimAx) +            ///<
                       (rprimGx + dy).dot(rprimGx + dy); ///<

            Hx(0, 1) = rdualQx.dot(rdualAv + rdualGw) + ///<
                       rcentuy;                         ///<

            Hx(1, 1) = (rdualAv + rdualGw).dot(rdualAv + rdualGw) + ///<
                       (dw - du).dot(dw - du);                      ///<

            Hx(1, 0) = Hx(0, 1);
        }

        return 0.5 * (rdual0.squaredNorm() + rdual1.squaredNorm() + rprimA.squaredNorm() + rprimG.squaredNorm()) +
               rcentr;
    };

    const auto x0       = make_vector<scalar_t>(max_pstep, max_dstep);
    const auto [xx, fx] = newton_step(x0, vgrad, logger);

    // apply the change
    const auto xstep = xx(0);
    const auto ystep = xx(0);
    const auto ustep = xx(1);
    const auto vstep = xx(1);
    const auto wstep = xx(1);

    /* pre-defined step lengths
    const auto dstep = (m == 0) ? s : make_umax(u, du, 1.0 - s);
    const auto pstep = (m == 0) ? s : make_umax(y, dy, 1.0 - s);

    const auto xstep = pstep;
    const auto ystep = pstep;
    const auto ustep = dstep;
    const auto vstep = dstep;
    const auto wstep = dstep;*/

    m_x.segment(0, n) = x + xstep * dx;
    m_x.segment(n, m) = y + ystep * dy;
    m_u               = u + ustep * du;
    m_v.segment(0, p) = v + vstep * dv;
    m_v.segment(p, m) = w + wstep * dw;

    update_original();
    update_residual();

    return lsearch_stats_t{xstep, ystep, ustep, vstep, wstep, fx};
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
