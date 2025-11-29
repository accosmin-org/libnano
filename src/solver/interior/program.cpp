#include <solver/interior/program.h>
#include <solver/interior/util.h>

using namespace nano;

program_t::program_t(const linear_program_t& program, linear_constraints_t constraints, const vector_t& x0)
    : program_t(program, matrix_t{}, program.c(), std::move(constraints), x0)
{
}

program_t::program_t(const quadratic_program_t& program, linear_constraints_t constraints, const vector_t& x0)
    : program_t(program, program.Q(), program.c(), std::move(constraints), x0)
{
}

program_t::program_t(const function_t& function, matrix_t Q, vector_t c, linear_constraints_t constraints,
                     const vector_t& x0)
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
{
    [[maybe_unused]] const auto n = this->n();
    [[maybe_unused]] const auto m = this->m();
    [[maybe_unused]] const auto p = this->p();

    assert(m_Q.size() == 0 || m_Q.rows() == n);
    assert(m_Q.size() == 0 || m_Q.cols() == n);

    assert(m_c.size() == n);

    assert(m_A.rows() == p);
    assert(m_A.cols() == n);
    assert(m_b.size() == p);

    assert(m_G.rows() == m);
    assert(m_G.cols() == n);
    assert(m_h.size() == m);

    ::nano::modified_ruiz_equilibration(m_dQ, m_Q, m_c, m_dG, m_G, m_h, m_dA, m_A, m_b);

    // initialize: see (2), p. 613, u = -1 / (G * x - h) = 1 / y
    m_x.segment(0, n)         = x0.vector();
    m_x.segment(n, m).array() = (m_h - m_G * x0).array().abs().max(1.0);
    m_u.array()               = 1.0 / m_x.segment(n, m).array();

    // move towards the center of the feasibility set to improve convergence: see (1), p. 485
    update_residual(0.0);
    update_solver();

    solve();

    m_x.segment(n, m).array() = (m_x.segment(n, m).array() + m_dx.segment(n, m).array()).abs().max(1.0);
    m_u.array()               = (m_u.array() + m_du.array()).abs().max(1.0);

    update_original();
}

program_t::stats_t program_t::update(const scalar_t tau)
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

    auto stats = stats_t{};

    if (m > 0)
    {
        // TODO: single matrix decomposition for predictor and corrector steps
        // TODO: couple step lengths for primal and dual variables

        // predictor step
        update_residual(0.0);
        update_solver();

        stats.m_predictor_stats = solve();

        const auto dy_affine    = dy;
        const auto du_affine    = du;
        const auto alpha_affine = std::min(make_umax(u, du, 1.0), make_umax(y, dy, 1.0));
        const auto miu          = y.dot(u);
        const auto miu_affine   = (y + alpha_affine * dy).dot(u + alpha_affine * du);

        assert(std::isfinite(miu));
        assert(std::isfinite(miu_affine));

        // TODO: safeguard sigma within reasonable limits
        // TODO: configurable power
        stats.m_sigma = std::pow(miu_affine / miu, 3.0);

        // corrector step
        update_residual(stats.m_sigma);
        m_rcent.array() += dy_affine.array() * du_affine.array();

        stats.m_corrector_stats = solve();

        // line-search
        const auto pstep = make_umax(y, dy, tau);
        const auto dstep = make_umax(u, du, tau);
        std::tie(stats.m_pstep, stats.m_dstep, stats.m_valid) = lsearch(std::min(pstep, dstep), std::min(pstep, dstep));
    }

    else
    {
        // NB: no inequalities, solve the affine KKT system directly!
        update_residual(0.0);
        update_solver();

        stats.m_predictor_stats = solve();
        stats.m_pstep           = tau;
        stats.m_dstep           = tau;
        stats.m_valid           = true;
    }

    // update primal-dual variables
    m_x.segment(0, n) = x + stats.m_pstep * dx;
    m_x.segment(n, m) = y + stats.m_pstep * dy;
    m_u               = u + stats.m_dstep * du;
    m_v.segment(0, p) = v + stats.m_dstep * dv;
    m_v.segment(p, m) = w + stats.m_dstep * dw;

    update_original();

    // compute residual
    update_residual(0.0);
    stats.m_residual = m_rdual.squaredNorm() + m_rprim.squaredNorm() + y.dot(u);

    return stats;
}

program_t::kkt_stats_t program_t::solve()
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
    const auto a7 = -m_u.array() * b5.array() + y.array() * b2.array() + b3.array();

    // |Q     0       0     A^T   G^T|   |dxn|   |-rdn|
    // |0     0      -I      0     I |   |dxm|   |-rdm|
    // |0  diag(u) diag(y)   0     0 | * |du | = |-rc |
    // |A     0       0      0     0 |   |dvp|   |-rpp|
    // |G     I       0      0     0 |   |dvm|   |-rpm|

    m_lvec.segment(0, n) = b1 - m_G.transpose() * (a7.array() / y.array()).matrix();
    m_lvec.segment(n, p) = b4;

    m_lsol.vector() = m_solver.solve(m_lvec.vector());

    const auto dxn = m_lsol.segment(0, n);
    const auto dvp = m_lsol.segment(n, p);
    const auto dvm = a7.array() / y.array() + (m_u.array() / y.array()) * (m_G * dxn).array();

    m_dx.segment(0, n) = dxn;
    m_dx.segment(n, m) = b5 - m_G * dxn;
    m_du.segment(0, m) = dvm - b2.array();
    m_dv.segment(0, p) = dvp;
    m_dv.segment(p, m) = dvm;

    // verify solution
    const auto valid    = m_lmat.all_finite() && m_lvec.all_finite() && m_lsol.all_finite();
    const auto delta    = (m_lmat * m_lsol - m_lvec).lpNorm<Eigen::Infinity>();
    const auto rcond    = m_solver.rcond();
    const auto positive = m_solver.isPositive();
    const auto negative = m_solver.isNegative();

    return kkt_stats_t{delta, rcond, valid, positive, negative};
}

void program_t::update_solver()
{
    const auto n = this->n();
    const auto m = this->m();
    const auto p = this->p();

    const auto y = m_x.segment(n, m);

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

    m_solver.compute(m_lmat.matrix());
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

void program_t::update_residual(const scalar_t sigma)
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

        m_rcent.array() = u.array() * y.array() - sigma * y.dot(u) / static_cast<scalar_t>(m);
    }
}

std::tuple<scalar_t, scalar_t, bool> program_t::lsearch(const scalar_t pstep, const scalar_t dstep)
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

    auto xstep = pstep;
    auto ystep = pstep;
    auto ustep = dstep;
    auto vstep = dstep;
    auto wstep = dstep;

    const auto make_residual = [&]() -> scalar_t
    {
        auto residual = 0.0;

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

        residual += m_rdual.squaredNorm();

        // primal residual
        m_rprim.segment(0, p) = m_A * (x + xstep * dx) - m_b;
        m_rprim.segment(p, m) = m_G * (x + xstep * dx) + (y + ystep * dy) - m_h;

        residual += m_rprim.squaredNorm();

        // centering residual
        if (m > 0)
        {
            residual += (y + ystep * dy).dot(u + ustep * du);
        }

        return residual;
    };

    const auto residual0 = m_rdual.squaredNorm() + m_rprim.squaredNorm() + y.dot(u);
    const auto max_iters = 100;
    const auto beta      = 0.9;
    const auto alpha     = 1e-6;

    auto valid = false;
    auto stepX = 1.0;

    for (auto iter = 0; iter < max_iters; ++iter)
    {
        if (const auto residualX = make_residual(); residualX < (1.0 - stepX * alpha) * residual0)
        {
            valid = true;
            break;
        }

        stepX *= beta;
        xstep *= beta;
        ystep *= beta;
        ustep *= beta;
        vstep *= beta;
        wstep *= beta;
    }

    assert((y + ystep * dy).minCoeff() > 0.0);
    assert((u + ustep * du).minCoeff() > 0.0);

    return std::make_tuple(ystep, ustep, valid);
}
