#include <Eigen/Dense>
#include <nano/core/numeric.h>
#include <nano/linprog.h>
#include <nano/tensor/stack.h>

using namespace nano;
using namespace nano::linprog;

namespace
{
auto Zero(const tensor_size_t rows)
{
    return vector_t::Zero(rows);
}

auto Zero(const tensor_size_t rows, const tensor_size_t cols)
{
    return matrix_t::Zero(rows, cols);
}

auto Identity(const tensor_size_t rows, const tensor_size_t cols)
{
    return matrix_t::Identity(rows, cols);
}

auto make_starting_point(const linprog::problem_t& problem)
{
    // see ch.14 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
    const auto& c = problem.m_c;
    const auto& A = problem.m_A;
    const auto& b = problem.m_b;

    const auto de = (A * A.transpose()).ldlt();

    vector_t       x = A.transpose() * de.solve(b);
    const vector_t l = de.solve(A * c);
    vector_t       s = c - A.transpose() * l;

    if (x.minCoeff() >= 0.0 && s.minCoeff() >= 0.0)
    {
        return linprog::solution_t{x, l, s};
    }

    const auto epsilon = std::numeric_limits<scalar_t>::min();
    const auto delta_x = std::max(epsilon, -1.5 * x.minCoeff());
    const auto delta_s = std::max(epsilon, -1.5 * s.minCoeff());

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
        if (dv(i) < std::numeric_limits<scalar_t>::min())
        {
            min = std::min(min, -v(i) / dv(i));
        }
    }
    return min;
}

template <typename tvector>
auto make_miu(const tvector& x, const tvector& s)
{
    assert(x.size() == s.size());

    return x.dot(s) / static_cast<scalar_t>(x.size());
}

template <typename tvector>
auto make_kkt(const linprog::problem_t& problem, const tvector& x, const tvector& l, const tvector& s)
{
    // see ch.14 (page 394) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
    const auto test1 = (problem.m_A.transpose() * l + s - problem.m_c).array().abs().maxCoeff();
    const auto test2 = (problem.m_A * x - problem.m_b).array().abs().maxCoeff();
    const auto test3 = (s.array() * x.array()).abs().maxCoeff();
    const auto test4 = x.array().min(0.0).abs().maxCoeff();
    const auto test5 = s.array().min(0.0).abs().maxCoeff();
    return test1 + test2 + test3 + test4 + test5;
}

auto make_kkt(const linprog::problem_t& problem, const linprog::solution_t& solution)
{
    return make_kkt(problem, solution.m_x, solution.m_l, solution.m_s);
}

struct scratchpad_t
{
    explicit scratchpad_t(const matrix_t& A)
        : scratchpad_t(A.cols(), A.rows())
    {
    }

    scratchpad_t(const tensor_size_t n, const tensor_size_t m)
        : m_rb(m)
        , m_rc(n)
        , m_rxs(n)
        , m_dx(n)
        , m_dl(m)
        , m_ds(n)
        , m_mat(m, m)
        , m_vec(m)
    {
    }

    template <typename tdecomposition>
    void solve(const tdecomposition& decomposition, const matrix_t& A, const vector_t& x, const vector_t& s)
    {
        assert(A.rows() == m_dl.size());
        assert(A.cols() == m_dx.size());

        // see eq. 14.44 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
        m_vec = -m_rb - A * ((x.array() * m_rc.array() - m_rxs.array()) / s.array()).matrix();
        m_dl  = decomposition.solve(m_vec);
        m_ds  = -m_rc - A.transpose() * m_dl;
        m_dx  = -(m_rxs.array() + x.array() * m_ds.array()) / s.array();
    }

    // attributes
    vector_t m_rb;  ///<
    vector_t m_rc;  ///<
    vector_t m_rxs; ///<
    vector_t m_dx;  ///<
    vector_t m_dl;  ///<
    vector_t m_ds;  ///<
    matrix_t m_mat; ///< (m, m) buffer to solve the linear system
    vector_t m_vec; ///< (m) buffer to solve the linear system
};
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

    auto c = stack<scalar_t>(2 * n + m, m_c, -m_c, Zero(m));
    auto A = stack<scalar_t>(m, 2 * n + m, m_A, -m_A, Identity(m, m));

    return {std::move(c), std::move(A), m_b};
}

linprog::solution_t linprog::inequality_problem_t::transform(const solution_t& isolution) const
{
    const auto                  n = m_c.size();
    [[maybe_unused]] const auto m = m_b.size();

    assert(isolution.m_x.size() == 2 * n + m);
    assert(isolution.m_s.size() == 2 * n + m);

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

    auto c = stack<scalar_t>(2 * n + m2, m_c, -m_c, Zero(m2));
    auto A = stack<scalar_t>(m1 + m2, 2 * n + m2, m_A, -m_A, Zero(m1, m2), m_G, -m_G, Identity(m2, m2));

    auto b = stack<scalar_t>(m1 + m2, m_b, m_h);

    return {std::move(c), std::move(A), std::move(b)};
}

linprog::solution_t linprog::general_problem_t::transform(const solution_t& isolution) const
{
    const auto                  n  = m_c.size();
    [[maybe_unused]] const auto m1 = m_b.size();
    [[maybe_unused]] const auto m2 = m_h.size();

    assert(isolution.m_x.size() == 2 * n + m2);
    assert(isolution.m_s.size() == 2 * n + m2);

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

std::optional<std::pair<matrix_t, vector_t>> linprog::make_independant_equality_constraints(const matrix_t& A,
                                                                                            const vector_t& b)
{
    assert(A.rows() == b.size());

    const auto dd = A.transpose().fullPivLu();

    // independant linear constraints
    if (dd.rank() == A.rows())
    {
        return std::nullopt;
    }

    // dependant linear constraints, use decomposition to formulate equivalent linear equality constraints
    const auto& P  = dd.permutationP();
    const auto& Q  = dd.permutationQ();
    const auto& LU = dd.matrixLU();

    const auto L = LU.leftCols(std::min(A.rows(), A.cols())).triangularView<Eigen::UnitLower>().toDenseMatrix();
    const auto U = LU.topRows(std::min(A.rows(), A.cols())).triangularView<Eigen::Upper>().toDenseMatrix();

    return std::make_pair(matrix_t{U.transpose().block(0, 0, dd.rank(), U.rows()) * L.transpose() * P},
                          vector_t{(Q.transpose() * b).segment(0, dd.rank())});
}

solver_t::solver_t(logger_t logger)
    : m_logger(std::move(logger))
{
    register_parameter(parameter_t::make_scalar("solver::epsilon", 0, LT, 1e-15, LE, 1e-3));
    register_parameter(parameter_t::make_integer("solver::max_iters", 10, LE, 100, LE, 1000));
    register_parameter(parameter_t::make_integer("solver::patience", 1, LE, 1, LE, 10));
    register_parameter(parameter_t::make_scalar("solver::eta0", 0.0, LT, 0.1, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::etaP", 1.0, LE, 3.0, LE, 9.0));
}

solution_t solver_t::solve(const problem_t& problem) const
{
    // NB: preprocess problem to remove dependant linear equality constraints!
    auto result = make_independant_equality_constraints(problem.m_A, problem.m_b);
    if (result)
    {
        auto& Ab = result.value();
        return solve_({problem.m_c, std::move(Ab.first), std::move(Ab.second)});
    }
    else
    {
        return solve_(problem);
    }
}

solution_t solver_t::solve_(const problem_t& problem) const
{
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();
    const auto max_iters = parameter("solver::max_iters").value<tensor_size_t>();
    const auto patience  = parameter("solver::patience").value<tensor_size_t>();
    const auto eta0      = parameter("solver::eta0").value<scalar_t>();
    const auto etaP      = parameter("solver::etaP").value<scalar_t>();

    auto cstate  = make_starting_point(problem); // current state
    auto bstate  = cstate;                       // best state wrt KKT violation
    auto scratch = scratchpad_t{problem.m_A};    // TODO: buffer to reuse if solving linear problems of the same size

    for (cstate.m_iters = 0; cstate.m_iters < max_iters; ++cstate.m_iters)
    {
        auto& x = cstate.m_x;
        auto& l = cstate.m_l;
        auto& s = cstate.m_s;

        // matrix decomposition to solve the linear systems (performed once per iteration)
        // see eq. 14.44 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
        scratch.m_mat = problem.m_A * (x.array() / s.array()).matrix().asDiagonal() * problem.m_A.transpose();

        // compute statistics
        const auto decomposition = scratch.m_mat.ldlt();
        cstate.m_miu             = make_miu(x, s);
        cstate.m_kkt             = make_kkt(problem, cstate);
        cstate.m_ldlt_rcond      = decomposition.rcond();
        cstate.m_ldlt_positive   = decomposition.isPositive();

        const auto valid = std::isfinite(cstate.m_miu) && std::isfinite(cstate.m_kkt) && x.allFinite();
        if (valid && cstate.m_kkt < bstate.m_kkt)
        {
            bstate = cstate;
        }
        if (m_logger)
        {
            m_logger(problem, cstate);
        }

        // check stopping criteria:
        // - divergence
        // - convergence
        // - no improvement in the past iterations
        if (!valid || bstate.converged(epsilon) || (cstate.m_iters >= patience + bstate.m_iters))
        {
            break;
        }

        // predictor step
        // see eq. 14.7-8 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
        scratch.m_rb  = problem.m_A * x - problem.m_b;
        scratch.m_rc  = problem.m_A.transpose() * l + s - problem.m_c;
        scratch.m_rxs = x.array() * s.array();
        scratch.solve(decomposition, problem.m_A, x, s);

        // centering step
        // see eq. 14.32-35 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
        const auto alpha_prim_aff = std::min(1.0, make_alpha(x, scratch.m_dx));
        const auto alpha_dual_aff = std::min(1.0, make_alpha(s, scratch.m_ds));

        const auto miu_aff = make_miu(x + alpha_prim_aff * scratch.m_dx, s + alpha_dual_aff * scratch.m_ds);
        const auto sigma   = cube(miu_aff / cstate.m_miu);

        scratch.m_rxs.array() += scratch.m_dx.array() * scratch.m_ds.array() - sigma * cstate.m_miu;
        scratch.solve(decomposition, problem.m_A, x, s);

        // update state
        const auto eta_base   = static_cast<scalar_t>(cstate.m_iters + 1);
        const auto eta        = 1.0 - eta0 / std::pow(eta_base, etaP);
        const auto alpha_prim = std::min(1.0, eta * make_alpha(x, scratch.m_dx));
        const auto alpha_dual = std::min(1.0, eta * make_alpha(s, scratch.m_ds));

        x += alpha_prim * scratch.m_dx;
        l += alpha_dual * scratch.m_dl;
        s += alpha_dual * scratch.m_ds;
    }

    return bstate;
}

solution_t solver_t::solve(const general_problem_t& problem) const
{
    const auto solution = solve(problem.transform());
    return problem.transform(solution);
}

solution_t solver_t::solve(const inequality_problem_t& problem) const
{
    const auto solution = solve(problem.transform());
    return problem.transform(solution);
}
