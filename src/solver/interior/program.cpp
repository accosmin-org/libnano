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
auto scale_ruiz(const matrix_t& A, const scalar_t epsilon = epsilon0<scalar_t>())
{
    auto Ak = A;
    auto D1 = make_full_vector<scalar_t>(A.rows(), 1.0);
    auto D2 = make_full_vector<scalar_t>(A.cols(), 1.0);
    auto Dr = vector_t{A.rows()};
    auto Dc = vector_t{A.cols()};

    for (auto k = 0; k < 100; ++k)
    {
        const auto critr = (1.0 - Dr.array()).matrix().lpNorm<Eigen::Infinity>();
        const auto critc = (1.0 - Dc.array()).matrix().lpNorm<Eigen::Infinity>();
        if (std::max({critr, critc}) < epsilon)
        {
            break;
        }

        Dr = Ak.matrix().rowwise().lpNorm<Eigen::Infinity>().array().sqrt();
        Dc = Ak.matrix().colwise().lpNorm<Eigen::Infinity>().array().sqrt();

        Ak = (1.0 / Dr.array()).matrix().asDiagonal() * Ak.matrix();
        Ak = Ak.matrix() * (1.0 / Dc.array()).matrix().asDiagonal();

        D1.array() /= Dr.array();
        D2.array() /= Dc.array();
    }

    return std::make_tuple(std::move(D1), std::move(Ak), std::move(D2));
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
    , m_lmat(n() + p(), n() + p())
    , m_lvec(n() + p())
    , m_lsol(n() + p())
{

    // fill the constant part of the matrix
    const auto n = this->n();
    const auto p = this->p();

    if (p > 0)
    {
        m_lmat.block(0, n, n, p) = m_A.transpose();
        m_lmat.block(n, 0, p, n) = m_A.matrix();
    }
    m_lmat.block(n, n, p, p).array() = 0.0;

    m_lsol.full(0.0);
}

const vector_t& program_t::solve() const
{
    /*// SCHUR complement approach
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
    const auto [D1, Ahat, D2] = ::scale_ruiz(m_lmat);

    m_ldlt.compute(Ahat.matrix());
    m_lsol.vector() = m_ldlt.solve((D1.array() * m_lvec.array()).matrix());
    m_lsol.array() *= D2.array();

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
    // solver.setTolerance(1e-10);
    // solver.compute(m_lmat.matrix());
    // m_lsol.vector() = solver.solve(m_lvec.vector());

    return m_lsol;
}

void program_t::update(const scalar_t ustep, const scalar_t xstep, const scalar_t miu, state_t& state) const
{
    const auto m = this->m();
    const auto p = this->p();

    state.m_x += xstep * state.m_dx;
    state.m_u += ustep * state.m_du;
    state.m_v += ustep * state.m_dv;

    // objective
    if (m_Q.size() == 0)
    {
        state.m_rdual = m_c;
    }
    else
    {
        state.m_rdual = Q() * state.m_x + m_c;
    }

    // surrogate duality gap
    if (m > 0)
    {
        state.m_eta = -state.m_u.dot(m_G * state.m_x - m_h);
    }

    // residual contributions of linear equality constraints
    if (p > 0)
    {
        state.m_rdual += m_A.transpose() * state.m_v;
        state.m_rprim = m_A * state.m_x - m_b;
    }

    // residual contributions of linear inequality constraints
    if (m > 0)
    {
        const auto sm = static_cast<scalar_t>(m);
        state.m_rdual += m_G.transpose() * state.m_u;
        state.m_rcent = -state.m_eta / (miu * sm) - state.m_u.array() * (m_G * state.m_x - m_h).array();
    }
}

bool program_t::valid(const scalar_t epsilon) const
{
    return m_lmat.all_finite() && m_lvec.all_finite() && m_lsol.all_finite() &&
           (m_lmat * m_lsol - m_lvec).lpNorm<Eigen::Infinity>() < epsilon;
}
