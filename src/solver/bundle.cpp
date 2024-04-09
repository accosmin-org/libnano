#include <iomanip>
#include <nano/core/logger.h>
#include <nano/solver/bundle.h>

using namespace nano;

template <>
nano::enum_map_t<bundle_pruning> nano::enum_string()
{
    return {
        { bundle_pruning::oldest_point,  "oldest"},
        {bundle_pruning::largest_error, "largest"},
    };
}

bundle_t::bundle_t(const solver_state_t& state, const tensor_size_t max_size, const bundle_pruning pruning)
    : m_bundleS(max_size + 1, state.x().size())
    , m_bundleE(max_size + 1)
    , m_alphas(max_size + 1)
    , m_x(state.x())
    , m_gx(state.gx())
    , m_fx(state.fx())
    , m_pruning(pruning)
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

void bundle_t::solve(const scalar_t miu)
{
    assert(size() > 0);
    assert(dims() == m_x.size());

    if (m_size == 1)
    {
        m_alphas(0) = 1.0;
    }
    else if (m_size == 2)
    {
        // NB: can compute analytically the solution for this case!
        const auto Q = S() * S().transpose();
        const auto c = miu * e();

        const auto q = Q(0, 0) + Q(1, 1) - Q(0, 1) - Q(1, 0);
        const auto p = 0.5 * (Q(0, 1) + Q(1, 0)) - Q(1, 1) + c(0) - c(1);

        const auto b = -p / q;
        const auto a = (std::isfinite(b) && b >= 0.0 && b <= 1.0) ? b : ((0.5 * q + p) > 0.0 ? 0.0 : 1.0);

        m_alphas(0) = a;
        m_alphas(1) = 1.0 - a;
    }
    else
    {
        const auto Q = S() * S().transpose();
        const auto c = miu * e();

        const auto lower = program::make_less(m_size, 1.0);
        const auto upper = program::make_greater(m_size, 0.0);
        const auto wsum1 = program::make_equality(vector_t::constant(m_size, 1.0), 1.0);

        const auto program = program::make_quadratic(Q, c, lower, upper, wsum1);

        const auto x0 = vector_t{vector_t::constant(m_size, 1.0 / static_cast<scalar_t>(m_size))};
        assert(program.feasible(x0, epsilon1<scalar_t>()));

        const auto solution = m_solver.solve(program, x0);
        debug_critical(!program.feasible(solution.m_x, epsilon1<scalar_t>()), std::fixed, std::setprecision(16),
                       "unfeasible solution to the bundle problem:\n\tQ=", Q, "\n\tc=", c,
                       "\n\tdeviation(eq)=", program.m_eq.deviation(solution.m_x),
                       "\n\tdeviation(ineq)=", program.m_ineq.deviation(solution.m_x));
        debug_critical(solution.m_status != solver_status::converged, std::fixed, std::setprecision(16),
                       "failed to solve the bundle problem:\n\tQ=", Q, "\n\tc=", c);

        m_alphas.slice(0, m_size) = solution.m_x;
    }
}

void bundle_t::delete_oldest(const tensor_size_t count)
{
    if (size() + 1 == capacity())
    {
        store_aggregate();

        assert(count <= size());
        m_size = remove_if([&](const tensor_size_t i) { return i < count; });

        append_aggregate();
    }
}

void bundle_t::delete_largest(const tensor_size_t count)
{
    if (size() + 1 == capacity())
    {
        store_aggregate();

        // NB: reuse the alphas buffer as it will be re-computed anyway at the next proximal point update!
        assert(count <= size());
        m_alphas.slice(0, size()) = m_bundleE.slice(0, size());
        std::nth_element(m_alphas.begin(), m_alphas.begin() + (size() - count), m_alphas.begin() + size());

        m_size = remove_if([&, thres = m_alphas(count) - epsilon0<scalar_t>()](const tensor_size_t i)
                           { return m_bundleE(i) > thres; });

        append_aggregate();
    }
}

void bundle_t::delete_inactive(const scalar_t epsilon)
{
    if (size() > 0)
    {
        m_size = remove_if([&](const tensor_size_t i) { return m_alphas(i) < epsilon; });
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
    // NB: the aggregation is stored in the last slot!
    const auto ilast         = capacity() - 1;
    m_bundleS.tensor(m_size) = m_bundleS.tensor(ilast);
    m_bundleE(m_size)        = m_bundleE(ilast);
    ++m_size;
}

void bundle_t::append(const vector_cmap_t y, const vector_cmap_t gy, const scalar_t fy, const bool serious_step)
{
    assert(dims() == y.size());
    assert(dims() == gy.size());

    delete_inactive(epsilon1<scalar_t>());
    if (m_pruning == bundle_pruning::oldest_point)
    {
        delete_oldest();
    }
    else
    {
        assert(m_pruning == bundle_pruning::largest_error);
        delete_largest();
    }

    // TODO: how to use the dual problem with a full quasi newton matrix (SR1) - see RQB reference

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
    c.register_parameter(parameter_t::make_enum(scat(prefix, "::bundle::pruning"), bundle_pruning::largest_error));
}

bundle_t bundle_t::make(const solver_state_t& state, const configurable_t& c, const string_t& prefix)
{
    const auto max_size = c.parameter(scat(prefix, "::bundle::max_size")).value<tensor_size_t>();
    const auto pruning  = c.parameter(scat(prefix, "::bundle::pruning")).value<bundle_pruning>();

    return {state, max_size, pruning};
}
