#include <solver/bundle.h>

using namespace nano;

namespace
{
auto make_program(const tensor_size_t n)
{
    return program::quadratic_program_t{matrix_t::zero(n + 1, n + 1), vector_t::zero(n + 1)};
}
} // namespace

bundle_t::solution_t::solution_t(const tensor_size_t dims)
    : m_x(dims)
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
    , m_bundleG(max_size + 1, state.x().size())
    , m_bundleF(max_size + 1)
    , m_optixr(state.x().size() + 1)
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

    const auto n = dims();
    const auto m = size();

    m_program.m_Q.block(n, n) = matrix_t::identity(n, n) / tau;
    m_program.m_c(n)          = 1.0;

    m_optixr.zero();
    m_optixr(n) = 1.0;

    m_program.constraint(program::make_less(bundleG(), bundleF()), program::make_less(m_optixr, level));

    const auto x0 = vector_t{vector_t::zero(n)};
    assert(program.feasible(x0, epsilon1<scalar_t>()));

    const auto solution = m_solver.solve(program, x0, logger);
    if (!program.feasible(solution.m_x, epsilon1<scalar_t>()))
    {
        logger.error(".bundle: unfeasible solution to the bundle problem:\n\tQ=", Q, "\n\tc=", c,
                     "\n\tdeviation(eq)=", program.m_eq.deviation(solution.m_x),
                     "\n\tdeviation(ineq)=", program.m_ineq.deviation(solution.m_x));
    }
    // TODO: the quadratic program may be unfeasible, so the level needs to moved towards the stability center
    if (solution.m_status != solver_status::converged)
    {
        logger.error(".bundle: failed to solve the bundle problem:\n\tQ=", Q, "\n\tc=", c);
    }

    // TODO: handle the case where the level is not given (!std::isfinite())

    m_solution.m_x = solution.m_x.segment(0, n) + m_x;
    m_solution.m_r = solution.m_x(n);
    assert(m_solution.m_r >= 0.0);

    m_solution.m_lambda = solution.m_u(n);
    assert(m_solution.m_lambda >= 0.0);

    const auto miu = solution.m_lambda + 1;

    m_solution.m_gnorm = ((m_x - m_solution.m_x) / (tau * miu)).lpNorm<2>();
    m_solution.m_epsil = (m_fx - m_solution.m_r) - (tau * miu) * square(m_solution.m_gnorm);

    return m_solution;
}

void bundle_t::delete_inactive(const scalar_t epsilon)
{
    if (size() > 0)
    {
        m_size = remove_if([&](const tensor_size_t i) { return m_alphas(i) < epsilon; });
    }
}

void bundle_t::delete_largest(const tensor_size_t count)
{
    if (size() + 1 == capacity())
    {
        store_aggregate();

        // NB: reuse the alphas buffer as it will be re-computed anyway at the next proximal point update!
        [[maybe_unused]] const auto old_size = m_size;
        assert(count <= m_size);

        m_alphas.slice(0, m_size) = m_bundleE.slice(0, m_size);
        std::nth_element(m_alphas.begin(), m_alphas.begin() + (m_size - count - 1), m_alphas.begin() + m_size);

        const auto threshold = m_alphas(m_size - count - 1);

        m_size = remove_if([&](const tensor_size_t i) { return m_bundleE(i) >= threshold; });

        assert(m_size + count <= old_size);
        assert(m_bundleE.slice(0, m_size).max() <= threshold + std::numeric_limits<scalar_t>::epsilon());

        append_aggregate();
    }
}

void bundle_t::store_aggregate()
{
    // NB: stored the aggregation in the last slot!
    const auto ilast        = capacity() - 1;
    m_bundleE(ilast)        = smeared_e();
    m_bundleS.tensor(ilast) = smeared_s();
}

void bundle_t::append_aggregate()
{
    // NB: load the aggregation from the last slot!
    const auto ilast         = capacity() - 1;
    m_bundleS.tensor(m_size) = m_bundleS.tensor(ilast);
    m_bundleE(m_size)        = m_bundleE(ilast);
    ++m_size;
}

void bundle_t::append(const vector_cmap_t y, const vector_cmap_t gy, const scalar_t fy, const bool serious_step)
{
    assert(dims() == y.size());
    assert(dims() == gy.size());

    delete_inactive(epsilon0<scalar_t>());
    delete_largest(2);

    if (serious_step)
    {
        for (tensor_size_t i = 0; i < m_size; ++i)
        {
            m_bundleE(i) += fy - m_fx - m_bundleS.vector(i).dot(y - m_x);
        }
        m_bundleE(m_size)        = 0.0;
        m_bundleS.tensor(m_size) = gy;
    }
    else
    {
        m_bundleE(m_size)        = m_fx - (fy + gy.dot(m_x - y));
        m_bundleS.tensor(m_size) = gy;
    }
    ++m_size;

    assert(m_size < capacity());
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

bool bundle_t::econverged(const scalar_t epsilon) const
{
    return smeared_e() <= epsilon;
}

bool bundle_t::sconverged(const scalar_t epsilon) const
{
    return smeared_s().template lpNorm<2>() <= epsilon;
}
