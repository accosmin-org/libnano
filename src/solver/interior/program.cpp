#include <Eigen/IterativeLinearSolvers>
#include <nano/function/util.h>
#include <solver/interior/minres.h>
#include <solver/interior/program.h>
#include <unsupported/Eigen/IterativeSolvers>

#include <iomanip>
#include <iostream>
#include <nano/function/lambda.h>
#include <nano/solver.h>

using namespace nano;

namespace
{
auto solve_kkt(matrix_t& lmat, vector_t& lvec, vector_t& lsol)
{
    // Ruiz-scale the system and solve it
    const auto& [D1, D2] = ::nano::scale_ruiz(lmat);
    lvec.array() *= D1.array();

    auto solver = Eigen::LDLT<eigen_matrix_t<scalar_t>>{lmat.matrix()};
    // auto solver = Eigen::BiCGSTAB<eigen_matrix_t<scalar_t>>{lmat.matrix()};
    // auto solver = Eigen::ConjugateGradient<eigen_matrix_t<scalar_t>>{lmat.matrix()};

    lsol.vector() = solver.solve(lvec.vector());
    lsol.array() *= D2.array();

    // verify solution
    const auto valid = lmat.all_finite() && lvec.all_finite() && lsol.all_finite();
    const auto delta = (lmat * lsol - lvec).lpNorm<Eigen::Infinity>();

    return program_t::solve_stats_t{valid, delta};
}
} // namespace

program_t::program_t(const linear_program_t& program, linear_constraints_t constraints)
    : program_t(program, matrix_t{}, program.c(), std::move(constraints))
{
}

program_t::program_t(const quadratic_program_t& program, linear_constraints_t constraints)
    : program_t(program, program.Q(), program.c(), std::move(constraints))
{
}

program_t::program_t(const function_t& function, matrix_t Q, vector_t c, linear_constraints_t constraints)
    : m_function(function)
    , m_Q(std::move(Q))
    , m_c(std::move(c))
    , m_A(std::move(constraints.m_A))
    , m_b(std::move(constraints.m_b))
    , m_G(std::move(constraints.m_G))
    , m_h(std::move(constraints.m_h))
    , m_x(vector_t::zero(n()))
    , m_u(vector_t::zero(m()))
    , m_v(vector_t::zero(p()))
    , m_dx(vector_t::zero(n()))
    , m_du(vector_t::zero(m()))
    , m_dv(vector_t::zero(p()))
    , m_rdual(n())
    , m_rcent(m())
    , m_rprim(p())
{
}

program_t::solve_stats_t program_t::solve()
{
    if (m_A.size() == 0 && m_G.size() > 0)
    {
        return solve_noA();
    }
    else if (m_G.size() == 0 && m_A.size() > 0)
    {
        return solve_noG();
    }
    else
    {
        return solve_wAG();
    }
}

program_t::solve_stats_t program_t::solve_noA()
{
    const auto n = this->n();

    auto lmat = matrix_t{n, n};
    auto lvec = vector_t{n};
    auto lsol = vector_t{n};

    const auto Gxmh  = m_G * m_x - m_h;
    const auto hess  = m_G.transpose() * (m_u.array() / Gxmh.array()).matrix().asDiagonal() * m_G.matrix();
    const auto rdual = m_rdual + m_G.transpose() * (m_rcent.array() / Gxmh.array()).matrix();

    if (m_Q.size() == 0)
    {
        lmat = -hess;
    }
    else
    {
        lmat = m_Q - hess;
    }
    lvec = -rdual;

    const auto stats = solve_kkt(lmat, lvec, lsol);

    m_dx = lsol.segment(0, n);
    m_du = (m_rcent.array() - m_u.array() * (m_G * m_dx).array()) / Gxmh.array();

    return {stats.m_valid && m_dx.all_finite() && m_du.all_finite() && m_dv.all_finite(), stats.m_precision};
}

program_t::solve_stats_t program_t::solve_noG()
{
    const auto n = this->n();
    const auto p = this->p();

    auto lmat = matrix_t{n + p, n + p};
    auto lvec = vector_t{n + p};
    auto lsol = vector_t{n + p};

    if (m_Q.size() == 0)
    {
        lmat.block(0, 0, n, n).array() = 0.0;
    }
    else
    {
        lmat.block(0, 0, n, n) = m_Q.matrix();
    }
    lmat.block(0, n, n, p)         = m_A.transpose();
    lmat.block(n, 0, p, n)         = m_A.matrix();
    lmat.block(n, n, p, p).array() = 0.0;

    lvec.segment(0, n) = -m_rdual;
    lvec.segment(n, p) = -m_rprim;

    const auto stats = solve_kkt(lmat, lvec, lsol);

    m_dx = lsol.segment(0, n);
    m_dv = lsol.segment(n, p);

    return {stats.m_valid && m_dx.all_finite() && m_du.all_finite() && m_dv.all_finite(), stats.m_precision};
}

program_t::solve_stats_t program_t::solve_wAG()
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

    // SCHUR complement approach
    /*{
        const auto n = this->n();
        const auto p = this->p();

        const auto A  = m_A.matrix();
        const auto H  = m_lmat.block(0, 0, n, n);
        const auto b1 = m_lvec.segment(0, n);
        const auto b2 = m_lvec.segment(n, p);

        // TODO: Ruiz scaling for H directly
        auto x1 = m_lsol.segment(0, n);
        auto x2 = m_lsol.segment(n, p);

        auto Hsolver = lin_solver_t{};
        Hsolver.compute(H);

        std::cout << std::setprecision(12) << "H =" << H << std::endl;
        std::cout << std::setprecision(12) << "b1=" << b1.transpose() << std::endl;
        std::cout << std::setprecision(12) << "b2=" << b2.transpose() << std::endl;
        std::cout << std::endl;

        if (p > 0)
        {
            const auto S = matrix_t{-A * Hsolver.solve(A.transpose())};

            auto xsolver = lin_solver_t{};
            xsolver.compute(S.matrix());

            x2 = xsolver.solve(b2 - A * Hsolver.solve(b1));
            x1 = Hsolver.solve(b1 - A.transpose() * x2);
        }
        else
        {
            x1 = Hsolver.solve(b1);
        }
    }*/

    /*{
        const auto [D1, Ahat, D2] = ::scale_ruiz(m_lmat);

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

    // Ruiz scaling algorithm that keeps the matrix symmetric
    // const auto [D1, Ahat, D2] = ::scale_ruiz(m_lmat);
    // m_ldlt.compute(Ahat.matrix());
    // m_lsol.vector() = m_ldlt.solve((D1.array() * m_lvec.array()).matrix());
    // m_lsol.array() *= D2.array();

    // LDLT (as positive semi-definite matrix)
    // m_ldlt.compute(m_lmat.matrix());
    // m_lsol.vector() = m_ldlt.solve(m_lvec.vector());

    // MINRES(m_lmat, m_lvec, m_lsol);
    // auto solver = Eigen::MINRES<eigen_matrix_t<scalar_t>, Eigen::Lower | Eigen::Upper,
    // Eigen::IdentityPreconditioner>{}; solver.compute(m_lmat.matrix()); m_lsol.vector() =
    // solver.solve(m_lvec.vector());

    // GMRES
    // auto solver = Eigen::GMRES<eigen_matrix_t<scalar_t>, Eigen::IdentityPreconditioner>{};
    // solver.setTolerance(1e-15);
    // solver.compute(m_lmat.matrix());
    // m_lsol.vector() = solver.solve(m_lvec.vector());

    // DGMRES
    // auto solver = Eigen::DGMRES<eigen_matrix_t<scalar_t>, Eigen::IdentityPreconditioner>{};
    // solver.setTolerance(1e-12);
    // solver.compute(m_lmat.matrix());
    // m_lsol.vector() = solver.solve(m_lvec.vector());

    // CG (as symmetric matrix)
    // auto solver = Eigen::ConjugateGradient<eigen_matrix_t<scalar_t>, Eigen::Lower | Eigen::Upper>{};
    // solver.setTolerance(1e-12);
    // solver.compute(m_lmat.matrix());
    // m_lsol.vector() = solver.solve(m_lvec.vector());

    // BiCBSTAB (as square matrix)
    // auto solver = Eigen::BiCGSTAB<eigen_matrix_t<scalar_t>>{};
    // solver.setTolerance(1e-14);
    // solver.compute(m_lmat.matrix());
    // m_lsol.vector() = solver.solve(m_lvec.vector());
}

scalar_t program_t::kkt_test() const
{
    return kkt_optimality_test(m_x, m_u, m_v);
}

scalar_t program_t::kkt_test(const scalar_t xstep, const scalar_t ustep, const scalar_t vstep) const
{
    return kkt_optimality_test(m_x + xstep * m_dx, m_u + ustep * m_du, m_v + vstep * m_dv);
}

void program_t::update(vector_t x, vector_t u, vector_t v, scalar_t miu)
{
    m_x = std::move(x);
    m_u = std::move(u);
    m_v = std::move(v);

    update(0.0, 0.0, 0.0, miu);
}

void program_t::update(const scalar_t xstep, const scalar_t ustep, const scalar_t vstep, const scalar_t miu)
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

    // apply the change
    m_x = x;
    m_u = u;
    m_v = v;
}
