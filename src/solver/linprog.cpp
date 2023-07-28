#include <Eigen/Dense>
#include <nano/core/numeric.h>
#include <nano/solver/linprog.h>

using namespace nano;

namespace
{
///
/// \brief return a starting point appropriate for primal-dual interior point methods.
///
/// see ch.14 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
auto make_starting_point(const linprog::problem_t& problem)
{
    const auto& c = problem.m_c;
    const auto& A = problem.m_A;
    const auto& b = problem.m_b;

    const matrix_t invA = (A * A.transpose()).inverse();

    vector_t       x = A.transpose() * invA * b;
    const vector_t l = invA * A * c;
    vector_t       s = c - A.transpose() * l;

    const auto delta_x = std::max(0.0, -1.5 * x.minCoeff());
    const auto delta_s = std::max(0.0, -1.5 * s.minCoeff());

    x.array() += delta_x;
    s.array() += delta_s;

    const auto delta_x_hat = 0.5 * x.dot(s) / s.sum();
    const auto delta_s_hat = 0.5 * x.dot(s) / x.sum();

    return linprog::solution_t{x.array() + delta_x_hat, l, s.array() + delta_s_hat};
}

template <typename tvector>
auto make_alpha(const vector_t& v, const tvector& dv)
{
    auto min = std::numeric_limits<scalar_t>::max();
    for (tensor_size_t i = 0, size = v.size(); i < size; ++i)
    {
        if (dv(i) < 0.0)
        {
            min = std::min(min, -v(i) / dv(i));
        }
    }
    return min;
}

template <typename tvector>
auto make_kkt(const linprog::problem_t& problem, const tvector& x, const tvector& l, const tvector& s)
{
    const auto test1 = (problem.m_A.transpose() * l + s - problem.m_c).array().abs().maxCoeff();
    const auto test2 = (problem.m_A * x - problem.m_b).array().abs().maxCoeff();
    const auto test3 = (s.array() * x.array()).abs().maxCoeff();
    const auto test4 = x.array().minCoeff();
    const auto test5 = s.array().minCoeff();
    return std::max(test1, std::max(std::max(test2, test3), std::max(test4, test5)));
}

auto make_kkt(const linprog::problem_t& problem, const linprog::solution_t& solution)
{
    return make_kkt(problem, solution.m_x, solution.m_l, solution.m_s);
}

auto make_eta(const int iters)
{
    return 1.0 - 0.1 / static_cast<scalar_t>(iters + 1);
    // const auto pow = std::pow(0.1, static_cast<double>(iters));
    // return std::max(0.9, 1.0 - pow + 0.5 * pow); // NB: 0.9, 0.95, 0.995, 0.9995, ...
}
} // namespace

linprog::problem_t::problem_t(vector_t c, matrix_t A, vector_t b)
    : m_c(std::move(c))
    , m_A(std::move(A))
    , m_b(std::move(b))
{
    assert(m_c.size() > 0);
    assert(m_b.size() > 0);
    assert(m_A.cols() == m_c.size());
    assert(m_A.rows() == m_b.size());
}

bool linprog::problem_t::feasible(const vector_t& x, const scalar_t epsilon) const
{
    assert(epsilon > 0.0);
    assert(x.size() == m_c.size());

    return (m_A * x - m_b).lpNorm<Eigen::Infinity>() < epsilon && x.minCoeff() >= 0.0;
}

linprog::inequality_problem_t::inequality_problem_t(vector_t c, matrix_t A, vector_t b)
    : m_c(std::move(c))
    , m_A(std::move(A))
    , m_b(std::move(b))
{
    assert(m_c.size() > 0);
    assert(m_b.size() > 0);
    assert(m_A.cols() == m_c.size());
    assert(m_A.rows() == m_b.size());
}

bool linprog::inequality_problem_t::feasible(const vector_t& x, const scalar_t epsilon) const
{
    assert(epsilon > 0.0);
    assert(x.size() == m_c.size());

    return (m_A * x - m_b).maxCoeff() < epsilon;
}

linprog::problem_t linprog::inequality_problem_t::transform() const
{
    const auto n = m_c.size();
    const auto m = m_b.size();

    auto c                      = vector_t{2 * n + m};
    c.segment(0, n)             = m_c;
    c.segment(n, n)             = -m_c;
    c.segment(2 * n, m).array() = 0;

    auto A              = matrix_t{m, 2 * n + m};
    A.block(0, 0, m, n) = m_A;
    A.block(0, n, m, n) = -m_A;
    A.block(0, 2 * n, m, m).setIdentity();

    return {std::move(c), std::move(A), m_b};
}

linprog::solution_t linprog::inequality_problem_t::transform(const solution_t& isolution) const
{
    const auto                  n = m_c.size();
    [[maybe_unused]] const auto m = m_b.size();

    assert(isolution.m_x.size() == 2 * n + m);
    assert(isolution.m_s.size() == 2 * n + m);
    assert(isolution.m_l.size() == m);

    auto solution        = isolution;
    solution.m_x         = isolution.m_x.segment(0, n) - isolution.m_x.segment(n, n);
    solution.m_s.array() = std::numeric_limits<scalar_t>::quiet_NaN(); // FIXME: double check this?!
    solution.m_l.array() = std::numeric_limits<scalar_t>::quiet_NaN(); // FIXME: double check this?!
    return solution;
}

linprog::general_problem_t::general_problem_t(vector_t c, matrix_t A, vector_t b, matrix_t G, vector_t h)
    : m_c(std::move(c))
    , m_A(std::move(A))
    , m_b(std::move(b))
    , m_G(std::move(G))
    , m_h(std::move(h))
{
    assert(m_c.size() > 0);
    assert(m_b.size() > 0);
    assert(m_A.cols() == m_c.size());
    assert(m_A.rows() == m_b.size());
    assert(m_G.cols() == m_c.size());
    assert(m_G.rows() == m_h.size());
}

bool linprog::general_problem_t::feasible(const vector_t& x, const scalar_t epsilon) const
{
    assert(epsilon > 0.0);
    assert(x.size() == m_c.size());

    return (m_A * x - m_b).lpNorm<Eigen::Infinity>() < epsilon && (m_G * x - m_h).maxCoeff() < epsilon;
}

linprog::problem_t linprog::general_problem_t::transform() const
{
    const auto n  = m_c.size();
    const auto m1 = m_b.size();
    const auto m2 = m_h.size();

    auto c                       = vector_t{2 * n + m2};
    c.segment(0, n)              = m_c;
    c.segment(n, n)              = -m_c;
    c.segment(2 * n, m2).array() = 0;

    auto A               = matrix_t{m1 + m2, 2 * n + m2};
    A.block(0, 0, m1, n) = m_A;
    A.block(0, n, m1, n) = -m_A;
    A.block(0, 2 * n, m2, m2).setZero();
    A.block(m1, 0, m2, n) = m_G;
    A.block(m1, n, m2, n) = -m_G;
    A.block(m1, 2 * n, m2, m2).setIdentity();

    auto b            = vector_t{m1 + m2};
    b.segment(0, m1)  = m_b;
    b.segment(m1, m2) = m_h;

    return {std::move(c), std::move(A), std::move(b)};
}

linprog::solution_t linprog::general_problem_t::transform(const solution_t& isolution) const
{
    const auto                  n  = m_c.size();
    [[maybe_unused]] const auto m1 = m_b.size();
    [[maybe_unused]] const auto m2 = m_h.size();

    assert(isolution.m_x.size() == 2 * n + m2);
    assert(isolution.m_s.size() == 2 * n + m2);
    assert(isolution.m_l.size() == m1 + m2);

    auto solution        = isolution;
    solution.m_x         = isolution.m_x.segment(0, n) - isolution.m_x.segment(n, n);
    solution.m_s.array() = std::numeric_limits<scalar_t>::quiet_NaN(); // FIXME: double check this?!
    solution.m_l.array() = std::numeric_limits<scalar_t>::quiet_NaN(); // FIXME: double check this?!
    return solution;
}

bool linprog::solution_t::converged(const scalar_t max_kkt_violation) const
{
    return std::isfinite(m_kkt) && m_kkt < max_kkt_violation;
}

linprog::solution_t linprog::solve(const problem_t& problem, const params_t& params)
{
    const auto& A = problem.m_A;
    const auto& b = problem.m_b;
    const auto& c = problem.m_c;

    const auto n = c.size();
    const auto m = A.rows();

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FIXME: these buffers can be allocated/stored once in a struct
    //  (if a linear problem of the same size needs to be solved many times)
    auto rb  = vector_t{m};
    auto rc  = vector_t{n};
    auto rxs = vector_t{n};
    auto dx  = vector_t{n};
    auto dl  = vector_t{m};
    auto ds  = vector_t{n};

    auto mat = matrix_t{m, m};                        // buffer to solve the linear system
    auto vec = vector_t{m};                           // buffer to solve the linear system
    auto mD2 = make_full_matrix<scalar_t>(n, n, 0.0); // D2 = diag(x/s)
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const auto solve = [&](const auto& decomposition, const vector_t& x, const vector_t& s)
    {
        // see eq. 14.44 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
        vec = -rb - A * ((x.array() * rc.array() - rxs.array()) / s.array()).matrix();
        dl  = decomposition.solve(vec);
        ds  = -rc - A.transpose() * dl;
        dx  = -(rxs.array() + x.array() * ds.array()) / s.array();
    };

    auto cstate = make_starting_point(problem); // current state
    auto bstate = cstate;                       // best state wrt KKT violation
    for (int eta_iters = 0; cstate.m_iters < params.m_max_iters && cstate.m_x.minCoeff() > 0.0; ++cstate.m_iters)
    {
        auto& x = cstate.m_x;
        auto& l = cstate.m_l;
        auto& s = cstate.m_s;

        // compute statistics
        cstate.m_miu = x.dot(s) / static_cast<scalar_t>(n);
        cstate.m_kkt = make_kkt(problem, cstate);

        const auto valid = std::isfinite(cstate.m_miu) && std::isfinite(cstate.m_kkt);
        if (valid && std::isfinite(cstate.m_kkt) && cstate.m_kkt < bstate.m_kkt)
        {
            bstate = cstate;
        }
        if (params.m_logger)
        {
            params.m_logger(problem, cstate);
        }

        // check stopping criteria:
        // - divergence
        // - convergence
        // - no improvement in the past iterations
        if (!valid || bstate.converged(params.m_kkt_epsilon) ||
            (cstate.m_iters >= params.m_kkt_patience + bstate.m_iters))
        {
            break;
        }

        // matrix decomposition to solve the linear systems (performed once per iteration)
        // see eq. 14.44 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
        mD2.diagonal() = x.array() / s.array();
        mat            = A * mD2 * A.transpose();

        const auto decomposition = mat.ldlt();

        // predictor step
        rb  = A * x - b;
        rc  = A.transpose() * l + s - c;
        rxs = x.array() * s.array();
        solve(decomposition, x, s);

        // centering step
        const auto alpha_prim_aff = std::min(1.0, make_alpha(x, dx));
        const auto alpha_dual_aff = std::min(1.0, make_alpha(s, ds));

        const auto miu_aff = (x + alpha_prim_aff * dx).dot(s + alpha_dual_aff * ds) / static_cast<scalar_t>(n);
        const auto sigma   = cube(miu_aff / cstate.m_miu);

        rxs.array() += dx.array() * ds.array() - sigma * cstate.m_miu;
        solve(decomposition, x, s);

        // update state
        const auto eta        = make_eta(eta_iters++);
        const auto alpha_prim = std::min(1.0, eta * make_alpha(x, dx));
        const auto alpha_dual = std::min(1.0, eta * make_alpha(s, ds));

        x += alpha_prim * dx;
        l += alpha_dual * dl;
        s += alpha_dual * ds;
    }

    return bstate;
}

linprog::solution_t linprog::solve(const general_problem_t& problem, const params_t& params)
{
    const auto solution = solve(problem.transform(), params);
    return problem.transform(solution);
}

linprog::solution_t linprog::solve(const inequality_problem_t& problem, const params_t& params)
{
    const auto solution = solve(problem.transform(), params);
    return problem.transform(solution);
}

linprog::params_t linprog::make_params(logger_t logger)
{
    auto params     = params_t{};
    params.m_logger = std::move(logger);
    return params;
}
