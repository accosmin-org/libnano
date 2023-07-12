#include <Eigen/Dense>
#include <nano/core/numeric.h>
#include <nano/solver/linprog.h>

using namespace nano;

namespace
{
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
    assert(x.size() == m_c.size());
    return (m_A * x - m_b).lpNorm<Eigen::Infinity>() < epsilon && x.minCoeff() >= 0.0;
}

bool linprog::solution_t::converged() const
{
    return m_miu < std::numeric_limits<scalar_t>::epsilon();
}

bool linprog::solution_t::diverged() const
{
    return !std::isfinite(m_miu) || m_miu > 1e+10;
}

linprog::solution_t linprog::make_starting_point(const linprog::problem_t& prog)
{
    const auto& c = prog.m_c;
    const auto& A = prog.m_A;
    const auto& b = prog.m_b;

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

    return {x.array() + delta_x_hat, l, s.array() + delta_s_hat};
}

linprog::solution_t linprog::solve(const linprog::problem_t& prog, const linprog::logger_t& logger)
{
    const auto& A = prog.m_A;
    const auto& b = prog.m_b;
    const auto& c = prog.m_c;

    const auto n = c.size();
    const auto m = A.rows();

    const auto max_iters = 100;

    auto rb  = vector_t{m};
    auto rc  = vector_t{n};
    auto mat = matrix_t{2 * n + m, 2 * n + m};
    auto vec = vector_t{2 * n + m};
    auto sol = vector_t{2 * n + m};

    mat.array()           = 0.0;
    mat.block(n, 0, m, n) = A;
    mat.block(0, n, n, m) = A.transpose();
    mat.block(0, n + m, n, n).setIdentity();

    const auto update_mat = [&](const vector_t& x, const vector_t& s)
    {
        mat.block(n + m, 0, n, n).diagonal()     = s;
        mat.block(n + m, n + m, n, n).diagonal() = x;
    };

    const auto solve = [&]()
    {
        sol = mat.lu().solve(vec);
        return std::make_tuple(sol.segment(0, n), sol.segment(n, m), sol.segment(n + m, n));
    };

    auto solution = make_starting_point(prog);
    for (; solution.m_iters < max_iters; ++solution.m_iters)
    {
        auto& x = solution.m_x;
        auto& l = solution.m_l;
        auto& s = solution.m_s;

        solution.m_miu = x.dot(s) / static_cast<scalar_t>(n);
        if (logger)
        {
            logger(solution);
        }
        if (solution.diverged() || solution.converged())
        {
            break;
        }

        update_mat(x, s);

        rb = A * x - b;
        rc = A.transpose() * l + s - c;

        vec.segment(0, n)             = -rc;
        vec.segment(n, m)             = -rb;
        vec.segment(n + m, n).array() = -x.array() * s.array();

        const auto [dx_aff, dl_aff, ds_aff] = solve();
        const auto alpha_pri_aff            = std::min(1.0, make_alpha(x, dx_aff));
        const auto alpha_dual_aff           = std::min(1.0, make_alpha(s, ds_aff));

        const auto miu_aff = (x + alpha_pri_aff * dx_aff).dot(s + alpha_dual_aff * ds_aff) / static_cast<scalar_t>(n);
        const auto sigma   = cube(miu_aff / solution.m_miu);

        vec.segment(n + m, n).array() -= dx_aff.array() * ds_aff.array() - sigma * solution.m_miu;

        const auto [dx, dl, ds] = solve();
        const auto eta          = 1.0 - std::pow(0.1, solution.m_iters + 1);
        const auto alpha_pri    = std::min(1.0, eta * make_alpha(x, dx));
        const auto alpha_dual   = std::min(1.0, eta * make_alpha(s, ds));

        x += alpha_pri * dx;
        l += alpha_dual * dl;
        s += alpha_dual * ds;
    }

    return solution;
}
