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
    const auto valid = lmat.all_finite() && lvec.all_finite() && lsol.all_finite();
    const auto delta = (lmat * lsol - lvec).lpNorm<Eigen::Infinity>();

    return program_t::solve_stats_t{valid, delta};
}
} // namespace

program_t::program_t(const linear_program_t& program, linear_constraints_t constraints, const scale_type scale,
                     const scalar_t miu)
    : program_t(program, matrix_t{}, program.c(), std::move(constraints), scale, miu)
{
}

program_t::program_t(const quadratic_program_t& program, linear_constraints_t constraints, const scale_type scale,
                     const scalar_t miu)
    : program_t(program, program.Q(), program.c(), std::move(constraints), scale, miu)
{
}

program_t::program_t(const function_t& function, matrix_t Q, vector_t c, linear_constraints_t constraints,
                     const scale_type scale, const scalar_t miu)
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

    m_x.segment(n(), m()).array() = 1.0;
    m_u.array()                   = 1.0;

    update(0.0, 0.0, 0.0, miu);
}

program_t::solve_stats_t program_t::solve()
{
    const auto n = this->n();
    const auto m = this->m();
    const auto p = this->p();

    //  Q' = |Q 0|, c' = |c|
    //       |0 0|       |0|
    //
    //  A' = |A 0|, b' = |b|
    //       |G I|       |h|
    //
    //  G' = |0 -I|, h' = 0

    // FIXME: re-use the allocated matrices
    // FIXME: check if possible to solve smaller systems

    const auto Qp =
        (m == 0) ? m_Q : nano::stack<scalar_t>(n + m, n + m, m_Q, matrix_t::zero(n, m), matrix_t::zero(m, n + m));
    const auto cp = (m == 0) ? m_c : nano::stack<scalar_t>(n + m, m_c, vector_t::zero(m));

    const auto Ap = (m == 0) ? m_A
                  : (p == 0)
                      ? nano::stack<scalar_t>(m, n + m, m_G, matrix_t::identity(m, m))
                      : nano::stack<scalar_t>(p + m, n + m, m_A, matrix_t::zero(m, m), m_G, matrix_t::identity(m, m));
    const auto bp = (m == 0) ? m_b : nano::stack<scalar_t>(p + m, m_b, m_h);

    const auto Gp =
        (m == 0) ? matrix_t{0, n} : nano::stack<scalar_t>(m, n + m, matrix_t::zero(m, n), -matrix_t::identity(m, m));
    const auto hp = vector_t::zero(m);

    const auto Gxmh  = Gp * m_x - hp;
    const auto hess  = Gp.transpose() * (m_u.array() / Gxmh.array()).matrix().asDiagonal() * Gp.matrix();
    const auto rdual = m_rdual + Gp.transpose() * (m_rcent.array() / Gxmh.array()).matrix();

    auto lmat = matrix_t{n + 2 * m + p, n + 2 * m + p};
    auto lvec = vector_t{n + 2 * m + p};
    auto lsol = vector_t{n + 2 * m + p};

    if (Qp.size() == 0)
    {
        lmat.block(0, 0, n + m, n + m) = -hess;
    }
    else
    {
        lmat.block(0, 0, n + m, n + m) = Qp - hess;
    }
    lmat.block(0, n + m, n + m, p + m)             = Ap.transpose();
    lmat.block(n + m, 0, p + m, n + m)             = Ap.matrix();
    lmat.block(n + m, n + m, p + m, p + m).array() = 0.0;

    lvec.segment(0, n + m)     = -rdual;
    lvec.segment(n + m, p + m) = -m_rprim;

    const auto stats = solve_kkt(lmat, lvec, lsol);

    m_dx = lsol.segment(0, n + m);
    m_dv = lsol.segment(n + m, p + m);
    m_du = (m_rcent.array() - m_u.array() * (Gp * m_dx).array()) / Gxmh.array();

    return {stats.m_valid && m_dx.all_finite() && m_du.all_finite() && m_dv.all_finite(), stats.m_precision};
}

scalar_t program_t::residual() const
{
    return std::sqrt(m_rdual.squaredNorm() + m_rcent.squaredNorm() + m_rprim.squaredNorm());
}

scalar_t program_t::update(const scalar_t xstep, const scalar_t ustep, const scalar_t vstep, const scalar_t miu,
                           const bool apply)
{
    const auto n = this->n();
    const auto m = this->m();
    const auto p = this->p();

    const auto x = m_x + xstep * m_dx;
    const auto u = m_u + ustep * m_du;
    const auto v = m_v + vstep * m_dv;

    const auto Qp =
        (m == 0) ? m_Q : nano::stack<scalar_t>(n + m, n + m, m_Q, matrix_t::zero(n, m), matrix_t::zero(m, n + m));
    const auto cp = (m == 0) ? m_c : nano::stack<scalar_t>(n + m, m_c, vector_t::zero(m));

    const auto Ap = (m == 0) ? m_A
                  : (p == 0)
                      ? nano::stack<scalar_t>(m, n + m, m_G, matrix_t::identity(m, m))
                      : nano::stack<scalar_t>(p + m, n + m, m_A, matrix_t::zero(m, m), m_G, matrix_t::identity(m, m));
    const auto bp = (m == 0) ? m_b : nano::stack<scalar_t>(p + m, m_b, m_h);

    const auto Gp =
        (m == 0) ? matrix_t{0, n} : nano::stack<scalar_t>(m, n + m, matrix_t::zero(m, n), -matrix_t::identity(m, m));
    const auto hp = vector_t::zero(m);

    // objective
    if (Qp.size() == 0)
    {
        m_rdual = cp;
    }
    else
    {
        m_rdual = Qp * x + cp;
    }

    // surrogate duality gap
    const auto eta = (m > 0) ? scalar_t{-u.dot(Gp * x - hp)} : 0.0;

    // residual contributions of linear equality constraints
    if (Ap.size() > 0)
    {
        m_rdual += Ap.transpose() * v;
        m_rprim = Ap * x - bp;
    }

    // residual contributions of linear inequality constraints
    if (Gp.size() > 0)
    {
        const auto sm = static_cast<scalar_t>(p + m);
        m_rdual += Gp.transpose() * u;
        m_rcent = -eta / (miu * sm) - u.array() * (Gp * x - hp).array();
    }

    // apply the change if requested
    if (apply)
    {
        m_x = x;
        m_u = u;
        m_v = v;

        update_original();
    }

    return residual();
}

scalar_t program_t::max_xstep() const
{
    const auto n = this->n();
    const auto m = this->m();

    return (m == 0) ? 1.0 : make_umax(m_x.segment(n, m), m_dx.segment(n, m));
}

scalar_t program_t::max_ustep() const
{
    const auto m = this->m();

    return (m == 0) ? 1.0 : make_umax(m_u, m_du);
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
