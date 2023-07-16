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

bool linprog::solution_t::converged(const scalar_t max_duality_measure) const
{
    return m_miu < max_duality_measure;
}

bool linprog::solution_t::diverged(const scalar_t min_duality_measure) const
{
    return !std::isfinite(m_miu) || m_miu > min_duality_measure;
}

linprog::solution_t linprog::solve(const linprog::problem_t& problem, const linprog::logger_t& logger)
{
    const auto& A = problem.m_A;
    const auto& b = problem.m_b;
    const auto& c = problem.m_c;

    const auto n = c.size();
    const auto m = A.rows();

    const auto max_iters      = 100;
    const auto max_eta        = 1.0 - 1e-8;
    const auto step_max_iters = 10;
    const auto step_factor    = 0.99;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FIXME: these buffers can be allocated/stored once in a struct
    //  (if a linear problem of the same size needs to be solved many times)
    auto rb  = vector_t{m};
    auto rc  = vector_t{n};
    auto rxs = vector_t{n};
    auto dx  = vector_t{n};
    auto dl  = vector_t{m};
    auto ds  = vector_t{n};

    auto mat = matrix_t{m, m}; // buffer to solve the linear system
    auto vec = vector_t{m};    // buffer to solve the linear system
    auto diX = matrix_t{n, n}; // diag(x)
    auto diS = matrix_t{n, n}; // diag(s)^-1

    diX.setZero();
    diS.setZero();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const auto solve = [&](const vector_t& x, const vector_t& s)
    {
        // see eq. 14.44 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
        diX.diagonal() = x;
        diS.diagonal() = 1.0 / s.array();

        mat = A * diX * diS * A.transpose();
        vec = -rb - A * diX * diS * rc + A * diS * rxs;

        dl = mat.llt().solve(vec);
        ds = -rc - A.transpose() * dl;
        dx = (-rxs.array() - x.array() * ds.array()) / s.array();
    };

    auto solution = make_starting_point(problem);
    for (; solution.m_iters < max_iters && solution.m_x.minCoeff() > 0.0; ++solution.m_iters)
    {
        auto& x = solution.m_x;
        auto& l = solution.m_l;
        auto& s = solution.m_s;

        solution.m_miu = x.dot(s) / static_cast<scalar_t>(n);
        if (logger)
        {
            logger(problem, solution);
        }
        if (solution.diverged() || solution.converged())
        {
            break;
        }

        rb  = A * x - b;
        rc  = A.transpose() * l + s - c;
        rxs = x.array() * s.array();
        solve(x, s);

        const auto alpha_pri_aff  = std::min(1.0, make_alpha(x, dx));
        const auto alpha_dual_aff = std::min(1.0, make_alpha(s, ds));

        const auto miu_aff = (x + alpha_pri_aff * dx).dot(s + alpha_dual_aff * ds) / static_cast<scalar_t>(n);
        const auto sigma   = cube(miu_aff / solution.m_miu);

        rxs.array() += dx.array() * ds.array() - sigma * solution.m_miu;
        solve(x, s);

        // step length search:
        // - decrease geometrically the step length if the duality measure is not decreased
        auto eta  = std::min(1.0 - std::pow(0.1, static_cast<double>(solution.m_iters + 1)), max_eta);
        auto iter = 0;
        for (; iter < step_max_iters; ++iter)
        {
            const auto alpha_pri  = std::min(1.0, eta * make_alpha(x, dx));
            const auto alpha_dual = std::min(1.0, eta * make_alpha(s, ds));

            const auto new_miu = (x + alpha_pri * dx).dot(s + alpha_dual * ds) / static_cast<scalar_t>(n);
            if (new_miu > solution.m_miu)
            {
                eta *= step_factor;
            }
            else
            {
                x += alpha_pri * dx;
                l += alpha_dual * dl;
                s += alpha_dual * ds;
                break;
            }
        }

        // cannot find an appropriate step length: unbounded problem?!,
        if (iter == step_max_iters)
        {
            solution.m_miu = std::numeric_limits<scalar_t>::max();
            break;
        }
    }

    return solution;
}
