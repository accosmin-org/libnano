#include <Eigen/IterativeLinearSolvers>
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

    return program_t::kkt_stats_t{delta, rcond, valid, positive, negative};
}
} // namespace

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
    m_u.array()               = 1 / m_x.segment(n, m).array();

    // move towards the center of the feasibility set to improve convergence: see (1), p. 485
    update_residual(0.0);

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
        // predictor step
        update_residual(0.0);

        stats.m_predictor_stats = solve();

        const auto alpha_affine = std::min(make_umax(u, du, 1e-12), make_umax(y, dy, 1e-12));
        const auto miu          = y.dot(u);
        const auto miu_affine   = (y + alpha_affine * dy).dot(u + alpha_affine * du);

        // TODO: safeguard sigma within reasonable limits
        // TODO: configurable power
        stats.m_sigma = std::pow(miu_affine / miu, 3.0);

        // corrector step
        update_residual(stats.m_sigma);
        m_rcent.array() += dy.array() * du.array();

        stats.m_corrector_stats = solve();
        stats.m_alpha           = std::min(make_umax(u, du, tau), make_umax(y, dy, tau));
    }

    else
    {
        // NB: no inequalities, solve the affine KKT system
        update_residual(0.0);

        stats.m_predictor_stats = solve();
        stats.m_alpha           = tau;
    }

    // update primal-dual variables
    m_x.segment(0, n) = x + stats.m_alpha * dx;
    m_x.segment(n, m) = y + stats.m_alpha * dy;
    m_u               = u + stats.m_alpha * du;
    m_v.segment(0, p) = v + stats.m_alpha * dv;
    m_v.segment(p, m) = w + stats.m_alpha * dw;

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
