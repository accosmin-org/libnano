#include <solver/bundle.h>

using namespace nano;

namespace
{
auto make_program(const tensor_size_t n)
{
    auto Q = matrix_t{matrix_t::zero(n + 1, n + 1)};
    auto c = vector_t{vector_t::zero(n + 1)};
    return program::quadratic_program_t{std::move(Q), std::move(c)};
}

template <typename tvectory>
auto eval_cutting_planes(matrix_cmap_t G, vector_cmap_t h, const tvectory& y)
{
    auto value = std::numeric_limits<scalar_t>::lowest();
    for (tensor_size_t i = 0, size = h.size(); i < size; ++i)
    {
        value = std::max(value, h(i) + G.vector(i).dot(y));
    }
    return value;
}

void write_cutting_plane(vector_map_t g, scalar_t& h, const vector_t& x, const vector_t& y, const vector_t& gy,
                         const scalar_t fy)
{
    const auto n = x.size();
    assert(y.size() == n);
    assert(gy.size() == n);
    assert(g.size() == n + 1);

    h                       = fy + gy.dot(x - y);
    g.segment(0, n).array() = gy.array();
    g(n)                    = -1.0;
}
} // namespace

bundle_t::solution_t::solution_t(const tensor_size_t dims)
    : m_x(dims)
    , m_ghat(dims)
{
}

bool bundle_t::solution_t::epsil_converged(const scalar_t epsilon) const
{
    return m_epsil < epsilon * std::sqrt(static_cast<scalar_t>(m_x.size()));
}

bool bundle_t::solution_t::gnorm_converged(const scalar_t epsilon) const
{
    return m_gnorm < epsilon * std::sqrt(static_cast<scalar_t>(m_x.size()));
}

bundle_t::bundle_t(const solver_state_t& state, const tensor_size_t max_size)
    : m_program(make_program(state.x().size()))
    , m_bundleG(max_size + 1, state.x().size() + 1)
    , m_bundleH(max_size + 1)
    , m_solution(state.x().size())
    , m_x(state.x())
    , m_gx(state.gx())
    , m_fx(state.fx())
{
    append(state.x(), state.gx(), state.fx(), true);
}

void bundle_t::moveto(const vector_cmap_t y, const vector_cmap_t gy, const scalar_t fy)
{
    const auto serious_step = true;
    append(y, gy, fy, serious_step);
    m_x  = y;
    m_gx = gy;
    m_fx = fy;
}

void bundle_t::append(const vector_cmap_t y, const vector_cmap_t gy, const scalar_t fy)
{
    const auto serious_step = false;
    append(y, gy, fy, serious_step);
}

const bundle_t::solution_t& bundle_t::solve(const scalar_t tau, const scalar_t level, const logger_t& logger)
{
    assert(size() > 0);
    assert(dims() == m_x.size());

    const auto n         = dims();
    const auto m         = size();
    const auto has_level = std::isfinite(level);

    const auto bundleG = m_bundleG.slice(0, m);
    const auto bundleH = m_bundleH.slice(0, m);

    // construct quadratic programming problem
    // NB: equivalent and simpler problem is to solve for `y = x - x_k^`!
    m_program.m_Q.block(0, 0, n, n).diagonal().array() = 1.0 / tau;
    m_program.m_c(n)          = 1.0;

    if (has_level)
    {
        auto weights    = m_bundleG.vector(capacity() - 1);
        weights.array() = 0.0;
        weights(n)      = 1.0;
        m_program.constrain(program::make_inequality(bundleG, -bundleH), program::make_inequality(weights, level));
    }
    else
    {
        m_program.constrain(program::make_inequality(bundleG, -m_bundleH));
    }

    // solve for (y, r) => (x = y + x_k^, r)!
    const auto& x0 = m_x;
    assert(m_program.feasible(x0, epsilon1<scalar_t>()));

    const auto solution = m_solver.solve(m_program, x0, logger);
    if (!m_program.feasible(solution.m_x, epsilon1<scalar_t>()))
    {
        logger.error("bundle: unfeasible solution, deviation(ineq)=", m_program.m_ineq.deviation(solution.m_x), ".\n");
    }
    // NB: the quadratic program may be unfeasible, so the level needs to moved towards the stability center!
    if (solution.m_status != solver_status::converged && !has_level)
    {
        logger.error("bundle: failed to solve, status=", solution.m_status, ".\n");
    }

    // extract solution and statistics, see (1)
    const auto y   = solution.m_x.segment(0, n);
    m_solution.m_x = y + m_x;
    m_solution.m_r = has_level ? solution.m_x(n) : 0.0;
    assert(m_solution.m_r >= 0.0);

    m_solution.m_lambda = solution.m_u(n);
    assert(m_solution.m_lambda >= 0.0);

    const auto miu = m_solution.m_lambda + 1.0;

    m_solution.m_ghat = -y / (tau * miu);
    m_solution.m_fhat = eval_cutting_planes(bundleG, bundleH, y);
    assert(m_solution.m_fhat <= m_solution.m_r);

    m_solution.m_gnorm = m_solution.m_ghat.lpNorm<2>();
    m_solution.m_epsil = (m_fx - m_solution.m_r) - (tau * miu) * square(m_solution.m_gnorm);
    m_solution.m_delta = m_fx - (m_solution.m_fhat + y.squaredNorm() / (2.0 * tau));

    assert(m_solution.m_epsil >= 0.0);
    assert(m_solution.m_delta >= 0.0);

    return m_solution;
}

void bundle_t::delete_inactive(const scalar_t epsilon)
{
    if (size() > 0)
    {
        m_bsize = remove_if([&](const tensor_size_t i) { return m_alphas(i) < epsilon; });
    }
}

void bundle_t::delete_largest(const tensor_size_t count)
{
    if (size() + 1 == capacity())
    {
        store_aggregate();

        m_bsize = remove_if([&](const tensor_size_t i) { return i < count; });

        /* FIXME: what are the bundle indices with the largest error in the new formulation?!
        // NB: reuse the alphas buffer as it will be re-computed anyway at the next proximal point update!
        [maybe_unused]] const auto old_size = m_bsize;
        assert(count <= m_bsize);

        m_alphas.slice(0, m_bsize) = m_bundleE.slice(0, m_bsize);
        std::nth_element(m_alphas.begin(), m_alphas.begin() + (m_bsize - count - 1), m_alphas.begin() + m_bsize);

        const auto threshold = m_alphas(m_bsize - count - 1);

        m_bsize = remove_if([&](const tensor_size_t i) { return m_bundleE(i) >= threshold; });

        assert(m_bsize + count <= old_size);
        assert(m_bundleE.slice(0, m_bsize).max() <= threshold + std::numeric_limits<scalar_t>::epsilon()); */

        append_aggregate();
    }
}

void bundle_t::store_aggregate()
{
    // NB: stored the aggregation in the last slot!
    const auto ilast        = capacity() - 1;
    write_cutting_plane(m_bundleG.vector(ilast), m_bundleH(ilast), m_x, m_solution.m_x, m_solution.m_ghat,
                        m_solution.m_fhat);
}

void bundle_t::append_aggregate()
{
    // NB: load the aggregation from the last slot!
    const auto ilast          = capacity() - 1;
    m_bundleH(m_bsize)        = m_bundleH(ilast);
    m_bundleG.vector(m_bsize) = m_bundleG.vector(ilast);
    ++m_bsize;
}

void bundle_t::append(const vector_cmap_t y, const vector_cmap_t gy, const scalar_t fy, const bool serious_step)
{
    assert(dims() == y.size());
    assert(dims() == gy.size());

    delete_inactive(epsilon0<scalar_t>());
    delete_largest(2);

    for (tensor_size_t i = 0; serious_step && i < m_bsize; ++i)
    {
        m_bundleH(i) += m_bundleG.row(i).segment(0, dims()).dot(y - m_x);
    }

    write_cutting_plane(m_bundleG.vector(m_bsize), m_bundleH(m_bsize), m_x, y, gy, fy);
    ++m_bsize;

    assert(m_bsize < capacity());
}

void bundle_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(parameter_t::make_integer(scat(prefix, "::bundle::max_size"), 2, LE, 100, LE, 1000));
}

bundle_t bundle_t::make(const solver_state_t& state, const configurable_t& c, const string_t& prefix)
{
    const auto max_size = c.parameter(scat(prefix, "::bundle::max_size")).value<tensor_size_t>();

    return {state, max_size};
}
