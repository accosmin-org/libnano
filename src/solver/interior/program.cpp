#include <Eigen/IterativeLinearSolvers>
#include <nano/function/util.h>
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

program_t::program_t(const linear_program_t& program, linear_constraints_t constraints, const scale_type scale)
    : program_t(program, matrix_t{}, program.c(), std::move(constraints), scale)
{
}

program_t::program_t(const quadratic_program_t& program, linear_constraints_t constraints, const scale_type scale)
    : program_t(program, program.Q(), program.c(), std::move(constraints), scale)
{
}

program_t::program_t(const function_t& function, matrix_t Q, vector_t c, linear_constraints_t constraints,
                     const scale_type scale)
    : m_function(function)
    , m_Q(std::move(Q))
    , m_c(std::move(c))
    , m_G(std::move(constraints.m_G))
    , m_h(std::move(constraints.m_h))
    , m_A(std::move(constraints.m_A))
    , m_b(std::move(constraints.m_b))
    , m_x(vector_t::zero(n()))
    , m_u(vector_t::zero(m()))
    , m_v(vector_t::zero(p()))
    , m_dx(vector_t::zero(n()))
    , m_du(vector_t::zero(m()))
    , m_dv(vector_t::zero(p()))
    , m_dQ(vector_t::constant(n(), 1.0))
    , m_dG(vector_t::constant(m(), 1.0))
    , m_dA(vector_t::constant(p(), 1.0))
    , m_rdual(n())
    , m_rcent(m())
    , m_rprim(p())
    , m_orig_x(n())
    , m_orig_u(m())
    , m_orig_v(p())
{
    switch (scale)
    {
    case scale_type::ruiz:
        ::nano::modified_ruiz_equilibration(m_dQ, m_Q, m_c, m_dG, m_G, m_h, m_dA, m_A, m_b);
        break;

    default:
        break;
    }

    assert(m_A.rows() == p());
    assert(m_A.cols() == n());
    assert(m_b.size() == p());

    assert(m_G.rows() == m());
    assert(m_G.cols() == n());
    assert(m_h.size() == m());
}

program_t::solve_stats_t program_t::solve()
{
    const auto n = this->n();
    const auto p = this->p();

    auto lmat = matrix_t{n + p, n + p};
    auto lvec = vector_t{n + p};
    auto lsol = vector_t{n + p};

    const auto Gxmh  = m_G * m_x - m_h;
    const auto hess  = m_G.transpose() * (m_u.array() / Gxmh.array()).matrix().asDiagonal() * m_G.matrix();
    const auto rdual = m_rdual + m_G.transpose() * (m_rcent.array() / Gxmh.array()).matrix();

    if (!m_Q.size())
    {
        lmat.block(0, 0, n, n) = -hess;
    }
    else
    {
        lmat.block(0, 0, n, n) = m_Q - hess;
    }
    lmat.block(0, n, n, p)         = m_A.transpose();
    lmat.block(n, 0, p, n)         = m_A.matrix();
    lmat.block(n, n, p, p).array() = 0.0;

    lvec.segment(0, n) = -rdual;
    lvec.segment(n, p) = -m_rprim;

    const auto stats = solve_kkt(lmat, lvec, lsol);

    m_dx = lsol.segment(0, n);
    m_dv = lsol.segment(n, p);
    m_du = (m_rcent.array() - m_u.array() * (m_G * m_dx).array()) / Gxmh.array();

    return {stats.m_valid && m_dx.all_finite() && m_du.all_finite() && m_dv.all_finite(), stats.m_precision};
}

scalar_t program_t::residual() const
{
    return std::sqrt(m_rdual.squaredNorm() + m_rcent.squaredNorm() + m_rprim.squaredNorm());
}

void program_t::update(vector_t x, vector_t u, vector_t v, scalar_t miu)
{
    assert(x.size() == n());
    assert(u.size() == m());
    assert(v.size() == p());

    m_x = std::move(x);
    m_u = std::move(u);
    m_v = std::move(v);

    update_original();
    update(0.0, 0.0, 0.0, miu);
}

scalar_t program_t::update(const scalar_t xstep, const scalar_t ustep, const scalar_t vstep, const scalar_t miu,
                           const bool apply)
{
    const auto m = this->m();
    const auto p = this->p();
    const auto x = m_x + xstep * m_dx;
    const auto u = m_u + ustep * m_du;
    const auto v = m_v + vstep * m_dv;

    // objective
    if (m_Q.size() == 0)
    {
        m_rdual = m_c;
    }
    else
    {
        m_rdual = Q() * x + m_c;
    }

    // surrogate duality gap
    const auto eta = (m > 0) ? scalar_t{-u.dot(m_G * x - m_h)} : 0.0;

    // residual contributions of linear equality constraints
    if (p > 0)
    {
        m_rdual += m_A.transpose() * v;
        m_rprim = m_A * x - m_b;
    }

    // residual contributions of linear inequality constraints
    if (m > 0)
    {
        const auto sm = static_cast<scalar_t>(m);
        m_rdual += m_G.transpose() * u;
        m_rcent = -eta / (miu * sm) - u.array() * (m_G * x - m_h).array();
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

void program_t::update_original()
{
    m_orig_x.array() = m_dQ.array() * m_x.array();
    m_orig_u.array() = m_dG.array() * m_u.array();
    m_orig_v.array() = m_dA.array() * m_v.array();
}

vector_t program_t::x(const vector_t& original_x) const
{
    return original_x.array() / m_dQ.array();
}
