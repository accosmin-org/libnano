#include <nano/solver/proximity.h>

using namespace nano;

namespace
{
scalar_t make_miu0(const solver_state_t& state)
{
    // see (4)
    return 5.0 * state.gx().squaredNorm() / (std::abs(state.fx()) + epsilon1<scalar_t>());
}

template <typename tnu, typename txi>
scalar_t make_miu(const scalar_t miu, const scalar_t t, const tnu& nu, const txi& xi)
{
    // see (2)
    const auto u = xi + t / miu * nu;
    assert(nu.dot(u) >= 0.0);
    //  TODO: make it work for other quasi-newton updates (e.g. SR1, BFGS)

    // NB: no positive solution if the function to optimize is not strictly convex!
    if (nu.dot(u) > epsilon0<scalar_t>())
    {
        return nu.dot(nu) / nu.dot(u);
    }
    else
    {
        return miu;
    }
}
} // namespace

proximity_t::proximity_t(const solver_state_t& state, const scalar_t miu0_min, const scalar_t miu0_max)
    : m_miu(std::clamp(make_miu0(state), miu0_min, miu0_max))
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
    m_miu = ::make_miu(m_miu, t, gn1 - gn, xn1 - xn);
}

void proximity_t::update(const scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn,
                         const vector_t& gn1, const vector_t& Gn, const vector_t& Gn1)
{
    // see (2)
    m_miu = std::min({::make_miu(m_miu, t, gn1 - gn, xn1 - xn), ::make_miu(m_miu, t, gn1 - Gn, xn1 - xn),
                      ::make_miu(m_miu, t, Gn1 - Gn, xn1 - xn), ::make_miu(m_miu, t, Gn1 - gn, xn1 - xn)});
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
