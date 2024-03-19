#include <nano/solver/proximity.h>

using namespace nano;

namespace
{
scalar_t make_miu0(const solver_state_t& state)
{
    // see (4)
    return 5.0 * state.gx().squaredNorm() / (std::abs(state.fx()) + epsilon1<scalar_t>());
}

void update(const scalar_t miu, matrix_t& M, matrix_t& invM)
{
    M    = miu * matrix_t::identity(M.rows(), M.cols());
    invM = (1.0 / miu) * matrix_t::identity(M.rows(), M.cols());
}
} // namespace

proximity_t::proximity_t(const solver_state_t& state, const scalar_t miu0_min, const scalar_t miu0_max)
    : m_M(state.x().size(), state.x().size())
    , m_invM(state.x().size(), state.x().size())
{
    const auto miu = std::clamp(make_miu0(state), miu0_min, miu0_max);
    assert(std::isfinite(miu));
    assert(miu > 0.0);

    ::update(miu, m_M, m_invM);
}

void proximity_t::update(const scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn,
                         const vector_t& gn1)
{
    // TODO: make it work for other quasi-newton updates (e.g. SR1, BFGS)

    const auto nu = gn1 - gn;
    const auto xi = xn1 - xn;

    const auto u = xi + t * m_invM * nu;
    assert(nu.dot(u) >= 0.0);

    // NB: no positive solution if the function to optimize is not strictly convex!
    if (nu.dot(u) > epsilon0<scalar_t>())
    {
        const auto dg = nu;
        const auto dx = u;

        // m_invM += (dx - m_invM * dg) * (dx - m_invM * dg).transpose() / (dx - m_invM * dg).dot(dg);
        const auto I = matrix_t::identity(m_invM.rows(), m_invM.cols());

        m_invM = (I - dx * dg.transpose() / dx.dot(dg)) * m_invM * (I - dg * dx.transpose() / dx.dot(dg)) +
                 dx * dx.transpose() / dx.dot(dg);

        // const auto miu = nu.dot(nu) / nu.dot(u);
        // ::update(miu, m_M, m_invM);
    }
}

void proximity_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(parameter_t::make_scalar_pair(scat(prefix, "::miu0_range"), 0.0, LT, 1e-4, LT, 1e+4, LT, 1e6));
}

proximity_t proximity_t::make(const solver_state_t& state, const configurable_t& c, const string_t& prefix)
{
    const auto [miu0_min, miu0_max] = c.parameter(scat(prefix, "::miu0_range")).value_pair<scalar_t>();

    return {state, miu0_min, miu0_max};
}
