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
    m_u.array()                   = 1.0;

    update(0.0, 0.0, 0.0, miu, true);
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

    const auto stats = solve_kkt(m_lmat, m_lvec, m_lsol);

    const auto dxn = m_lsol.segment(0, n);
    const auto dvp = m_lsol.segment(n, p);
    const auto dvm = a7.array() / y.array() + (m_u.array() / y.array()) * (m_G * dxn).array();

    m_dx.segment(0, n) = dxn;
    m_dx.segment(n, m) = b5 - m_G * dxn;
    m_du.segment(0, m) = dvm - b2.array();
    m_dv.segment(0, p) = dvp;
    m_dv.segment(p, m) = dvm;

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

    const auto z = x.segment(0, n);
    const auto y = x.segment(n, m);

    // dual residual
    if (m_Q.size() == 0)
    {
        m_rdual.segment(0, n).matrix() = m_c.vector();
    }
    else
    {
        m_rdual.segment(0, n).matrix() = m_Q * z + m_c;
    }
    m_rdual.segment(0, n) += m_A.transpose() * v.segment(0, p);
    m_rdual.segment(0, n) += m_G.transpose() * v.segment(p, m);

    m_rdual.segment(n, m) = v.segment(p, m) - u;

    // primal residual
    m_rprim.segment(0, p) = m_A * z - m_b;
    m_rprim.segment(p, m) = m_G * z + y - m_h;

    // centering residual
    if (m > 0)
    {
        m_rcent.array() = -u.dot(y) / (miu * static_cast<scalar_t>(m)) + u.array() * y.array();
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
