#include <nano/solver/proximity.h>

using namespace nano;

namespace
{
scalar_t make_miu0(const solver_state_t& state)
{
    // see (4)
    return 5.0 * state.gx().squaredNorm() / (std::abs(state.fx()) + epsilon0<scalar_t>());
}

template <typename tnu, typename txi>
scalar_t make_miu(const scalar_t miu, const scalar_t t, const tnu& nu, const txi& xi, const scalar_t min_dot_nuv)
{
    // see (2): reversal poor man's quasi-newton formula.
    const auto u = xi + t / miu * nu;
    // assert(nu.dot(u) >= 0.0);
    //  TODO: make it work for other quasi-newton updates (e.g. SR1, BFGS)

    // NB: no positive solution if the function to optimize is not strictly convex!
    if (nu.dot(u) > min_dot_nuv)
    {
        return nu.dot(nu) / nu.dot(u);
    }
    else
    {
        return std::numeric_limits<scalar_t>::max();
    }
}
} // namespace

proximity_t::proximity_t(const solver_state_t& state, const scalar_t miu0_min, const scalar_t miu0_max,
                         const scalar_t min_dot_nuv)
    : m_miu(std::clamp(make_miu0(state), miu0_min, miu0_max))
    , m_min_dot_nuv(min_dot_nuv)
{
}

scalar_t proximity_t::miu() const
{
    assert(std::isfinite(m_miu));
    assert(m_miu > 0.0);
    return m_miu;
}

void proximity_t::update(const scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn,
                         const vector_t& gn1)
{
    const auto xi = xn1 - xn;
    const auto nu = gn1 - gn;
    if (const auto miu = ::make_miu(m_miu, t, nu, xi, m_min_dot_nuv); miu != std::numeric_limits<scalar_t>::max())
    {
        m_miu = miu;
    }
}

void proximity_t::update(const scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn,
                         const vector_t& gn1, const vector_t& Gn, const vector_t& Gn1)
{
    // see (2): the variation that gives the minimum proximity parameter.
    auto miu = std::numeric_limits<scalar_t>::max();
    for (const auto alpha1 : {0.0, 0.5, 1.0})
    {
        for (const auto alpha2 : {0.0, 0.5, 1.0})
        {
            const auto xi = xn1 - xn;
            const auto nu = alpha1 * gn1 + (1.0 - alpha1) * Gn1 - alpha2 * gn - (1.0 - alpha2) * Gn;

            miu = std::min(miu, ::make_miu(m_miu, t, nu, xi, m_min_dot_nuv));
        }
    }

    if (miu != std::numeric_limits<scalar_t>::max())
    {
        m_miu = miu;
    }
}

void proximity_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(
        parameter_t::make_scalar_pair(scat(prefix, "::prox::miu0_range"), 0.0, LT, 1e+2, LT, 1e+4, LT, 1e6));
    c.register_parameter(
        parameter_t::make_scalar(scat(prefix, "::prox::min_dot_nuv"), 0.0, LT, epsilon0<scalar_t>(), LT, 1.0));
}

proximity_t proximity_t::make(const solver_state_t& state, const configurable_t& c, const string_t& prefix)
{
    const auto [miu0_min, miu0_max] = c.parameter(scat(prefix, "::prox::miu0_range")).value_pair<scalar_t>();
    const auto min_dot_nuv          = c.parameter(scat(prefix, "::prox::min_dot_nuv")).value<scalar_t>();

    return {state, miu0_min, miu0_max, min_dot_nuv};
}
