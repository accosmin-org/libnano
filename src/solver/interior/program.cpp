#include <solver/interior/program.h>
#include <solver/interior/util.h>

using namespace nano;

program_t::program_t(const linear_program_t& program, linear_constraints_t constraints, const vector_t& x0)
    : program_t(program, matrix_t::zero(x0.size(), x0.size()), program.c(), std::move(constraints), x0)
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
    assert(m_Q.rows() == n());
    assert(m_Q.cols() == n());

    assert(m_c.size() == n());

    assert(m_A.rows() == p());
    assert(m_A.cols() == n());
    assert(m_b.size() == p());

    assert(m_G.rows() == m());
    assert(m_G.cols() == n());
    assert(m_h.size() == m());

    ::nano::modified_ruiz_equilibration(m_dQ, m_Q, m_c, m_dG, m_G, m_h, m_dA, m_A, m_b);

    // initialize: see (2), p. 613, u = -1 / (G * x - h) = 1 / y
    auto [x, y, u, v, w] = unpack_vars();
    [[maybe_unused]] auto [dx, dy, du, dv, dw] = unpack_delta();

    x.array() = x0.array();
    y.array() = (m_h - m_G * x0).array().abs().max(1.0);
    u.array() = 1.0 / y.array();
    v.array() = 1.0;
    w.array() = u.array();

    /*// move towards the center of the feasibility set to improve convergence: see (1), p. 485
    update_solver();
    update_residual(0.0);
    solve(logger_t{});

    y.array() = (y + dy).array().abs().max(1.0);
    u.array() = (u + du).array().abs().max(1.0);
    w.array() = u.array();*/

    update_original();
}

program_t::stats_t program_t::update(const scalar_t tau, const logger_t& logger)
{
    const auto [n, m, p]            = unpack_dims();
    auto [x, y, u, v, w]            = unpack_vars();
    const auto [dx, dy, du, dv, dw] = unpack_delta();

    auto stats = stats_t{};

    update_solver();

    if (m > 0)
    {
        // predictor step
        update_residual(0.0);
        stats.m_predictor_stats = solve(logger);

        if (!du.allFinite() || !dy.allFinite())
        {
            stats.m_valid = false;
            return stats;
        }

        const auto alpha_affine = std::min(make_umax(y, dy, 1.0), make_umax(u, du, 1.0));
        const auto miu          = y.dot(u);
        const auto miu_affine   = (y + alpha_affine * dy).dot(u + alpha_affine * du);

        stats.m_sigma = std::clamp(std::pow(miu_affine / miu, 3.0), 0.0, 1.0);

        // corrector step
        m_rcent.array() =
            u.array() * y.array() + dy.array() * du.array() - stats.m_sigma * miu / static_cast<scalar_t>(m);

        stats.m_corrector_stats = solve(logger);

        if (!du.allFinite() || !dy.allFinite())
        {
            stats.m_valid = false;
            return stats;
        }

        // line-search
        const auto pstep = make_umax(y, dy, tau);
        const auto dstep = make_umax(u, du, tau);

        // std::tie(stats.m_alpha, stats.m_valid) = lsearch(std::min(pstep, dstep));
        stats.m_alpha = std::min(pstep, dstep);
        stats.m_valid = true;
    }
    else
    {
        // NB: no inequalities, solve the affine KKT system directly!
        update_residual(0.0);
        stats.m_predictor_stats = solve(logger);

        stats.m_alpha = tau;
        stats.m_valid = true;
    }

    // update primal-dual variables
    x += stats.m_alpha * dx;
    y += stats.m_alpha * dy;
    u += stats.m_alpha * du;
    v += stats.m_alpha * dv;
    w += stats.m_alpha * dw;

    // update original un-scaled primal-dual variables
    update_original();

    // compute convergence criteria
    const auto normQ = m_dQ.lpNorm<Eigen::Infinity>();
    const auto normA = m_dA.lpNorm<Eigen::Infinity>();
    const auto normG = m_dG.lpNorm<Eigen::Infinity>();

    stats.m_primal_residual =                          ///<
        (m_A * x - m_b).lpNorm<Eigen::Infinity>() +    ///<
        (m_G * x + y - m_h).lpNorm<Eigen::Infinity>(); ///<

    stats.m_dual_residual =                                                                     ///<
        (m_Q * x + m_c + m_A.transpose() * v + m_G.transpose() * w).lpNorm<Eigen::Infinity>() + ///<
        (w - u).lpNorm<Eigen::Infinity>();                                                      ///<

    stats.m_duality_gap = std::fabs(                     ///<
        x.dot(m_Q * x + m_c) + m_b.dot(v) + m_h.dot(w)); ///<

    stats.m_primal_residual *= 1.0 + normA + normG;
    stats.m_dual_residual *= 1.0 + normQ + normA + normG;
    stats.m_duality_gap *= 1.0 + normQ + normA + normG;

    return stats;
}

void program_t::refine_solution(const logger_t& logger, const int refine_max_iters, const scalar_t refine_epsilon,
                                const int patience)
{
    auto solution      = vector_t{m_lsol.size()};
    auto residual      = vector_t{m_lsol.size()};
    auto correction    = vector_t{m_lsol.size()};
    auto best_solution = vector_t{m_lsol.size()};
    auto best_accuracy = std::numeric_limits<scalar_t>::max();

    solution.vector() = m_solver.solve(m_lvec.vector());

    for (auto iter = 0, best_last_iter = 0; iter < refine_max_iters; ++iter)
    {
        residual = m_lvec - m_lmat * solution;

        const auto accuracy = residual.lpNorm<2>();
        logger.info("kktrefine: iter=", iter, ",accuracy=", accuracy, ".\n");

        if (accuracy < best_accuracy)
        {
            best_accuracy = accuracy;
            best_solution = solution;
            best_last_iter = iter;
        }
        if (accuracy < refine_epsilon)
        {
            break;
        }

        ++best_last_iter;
        if (best_last_iter > patience)
        {
            break;
        }

        correction.vector() = m_solver.solve(residual.vector());
        if (!correction.all_finite())
        {
            break;
        }

        solution += correction;
    }

    m_lsol = best_solution;
}

program_t::kkt_stats_t program_t::solve(const logger_t& logger)
{
    const auto [n, m, p]       = unpack_dims();
    const auto [x, y, u, v, w] = unpack_vars();

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

    refine_solution(logger);

    const auto dx = m_lsol.segment(0, n);
    const auto dv = m_lsol.segment(n, p);
    const auto dw = a7.array() / y.array() + (u.array() / y.array()) * (m_G * dx).array();

    m_dx.segment(0, n) = dx;
    m_dx.segment(n, m) = b5 - m_G * dx;
    m_du.segment(0, m) = dw - b2.array();
    m_dv.segment(0, p) = dv;
    m_dv.segment(p, m) = dw;

    // verify solution
    const auto valid    = m_lmat.all_finite() && m_lvec.all_finite() && m_lsol.all_finite();
    const auto accuracy = (m_lmat * m_lsol - m_lvec).lpNorm<2>();
    const auto rcond    = m_solver.rcond();
    const auto positive = m_solver.isPositive();
    const auto negative = m_solver.isNegative();

    return kkt_stats_t{
        .m_accuracy = accuracy, .m_rcond = rcond, .m_valid = valid, .m_positive = positive, .m_negative = negative};
}

void program_t::update_solver()
{
    const auto [n, m, p]       = unpack_dims();
    const auto [x, y, u, v, w] = unpack_vars();

    // |Q     0       0     A^T   G^T|   |dxn|   |-rdn|
    // |0     0      -I      0     I |   |dxm|   |-rdm|
    // |0  diag(u) diag(y)   0     0 | * |du | = |-rc |
    // |A     0       0      0     0 |   |dvp|   |-rpp|
    // |G     I       0      0     0 |   |dvm|   |-rpm|

    m_lmat.block(0, 0, n, n) = m_Q.matrix() + m_G.transpose() * (u.array() / y.array()).matrix().asDiagonal() * m_G;
    m_lmat.block(0, n, n, p) = m_A.transpose();
    m_lmat.block(n, 0, p, n) = m_A.matrix();
    m_lmat.block(n, n, p, p) = matrix_t::zero(p, p);

    m_solver.compute(m_lmat.matrix());
}

void program_t::update_original()
{
    const auto [x, y, u, v, w] = unpack_vars();

    m_orig_x.array() = m_dQ.array() * x.array();
    m_orig_u.array() = m_dG.array() * u.array();
    m_orig_v.array() = m_dA.array() * v.array();
}

void program_t::update_residual(const scalar_t sigma)
{
    const auto [n, m, p]       = unpack_dims();
    const auto [x, y, u, v, w] = unpack_vars();

    // dual residual
    m_rdual.segment(0, n).matrix() = m_Q * x + m_c;
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

scalar_t program_t::residual(const scalar_t lstep)
{
    const auto [n, m, p]            = unpack_dims();
    const auto [x, y, u, v, w]      = unpack_vars();
    const auto [dx, dy, du, dv, dw] = unpack_delta();

    auto residual = 0.0;

    // dual residual
    m_rdual.segment(0, n).matrix() = m_Q * (x + lstep * dx) + m_c;
    m_rdual.segment(0, n) += m_A.transpose() * (v + lstep * dv);
    m_rdual.segment(0, n) += m_G.transpose() * (w + lstep * dw);
    m_rdual.segment(n, m) = (w + lstep * dw) - (u + lstep * du);

    residual += m_rdual.lpNorm<2>() / (1.0 + m_c.lpNorm<2>());

    // primal residual
    m_rprim.segment(0, p) = m_A * (x + lstep * dx) - m_b;

    residual += m_rprim.segment(0, p).lpNorm<2>() / (1.0 + m_b.lpNorm<2>());

    m_rprim.segment(p, m) = m_G * (x + lstep * dx) + (y + lstep * dy) - m_h;

    residual += m_rprim.segment(p, m).lpNorm<2>() / (1.0 + m_h.lpNorm<2>());

    // centering residual
    if (m > 0)
    {
        assert((y + lstep * dy).minCoeff() > 0.0);
        assert((u + lstep * du).minCoeff() > 0.0);

        residual += std::sqrt((y + lstep * dy).dot(u + lstep * du));
    }

    return residual;
}

std::tuple<scalar_t, bool> program_t::lsearch(const scalar_t lstep0)
{
    const auto residual0 = residual(0.0);

    const auto max_iters = 100;
    const auto beta      = 0.9;
    const auto alpha     = 1e-6;

    auto valid = false;
    auto lstep = lstep0;

    for (auto iter = 0; iter < max_iters; ++iter)
    {
        if (const auto residualX = residual(lstep); residualX < (1.0 - lstep * alpha) * residual0)
        {
            valid = true;
            break;
        }

        lstep *= beta;
    }

    return std::make_tuple(lstep, valid);
}
