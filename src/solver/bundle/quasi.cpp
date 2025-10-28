#include <solver/bundle/quasi.h>

using namespace nano;
using namespace nano::bundle;

namespace{
scalar_t make_tau0(const solver_state_t& state, const scalar_t tau_min)
{
    // see (6)
    const auto tau0 = std::max(1.0, std::fabs(state.fx())) / (5.0 * state.gx().squaredNorm());
    return std::isfinite(tau0) ? std::max(tau0, tau_min) : tau_min;
}
} // namespace

namespace nano
{
template <>
enum_map_t<quasi_type> enum_string<quasi_type>()
{
    return {
        {quasi_type::sr1, "sr1"},
        {quasi_type::miu, "miu"}
    };
}
} // namespace nano

quasi_t::quasi_t(const solver_state_t& state, const quasi_type type, const scalar_t tau_min)
    : m_M(matrix_t::identity(state.x().size(), state.x().size()) * make_tau0(state, tau_min))
    , m_xn(state.x())
    , m_xn1(state.x())
    , m_gn(state.gx())
    , m_gn1(state.gx())
    , m_Gn(state.gx())
    , m_Gn1(state.gx())
    , m_type(type)
{
}

const matrix_t& quasi_t::update(const vector_t& x, const vector_t& g, const vector_t& G, const bool is_descent_step)
{
    assert(x.size() == m_xn.size());
    assert(g.size() == m_gn.size());
    assert(G.size() == m_Gn.size());

    m_xn = m_xn1;
    m_gn = m_gn1;
    m_Gn = m_Gn1;

    m_xn1 = x;
    m_gn1 = g;
    m_Gn1 = G;

    if (is_descent_step)
    {
        switch (m_type)
        {
        case quasi_type::miu:
            update_miu();
            break;

        default:
            update_sr1();
            break;
        }
    }

    return m_M;
}

void quasi_t::update_miu()
{
    // see (3)
    const auto e = m_xn1 - m_xn;
    const auto v1 = m_Gn1 - m_Gn;
    const auto v2 = m_Gn1 - m_gn;
    const auto v3 = m_gn1 - m_Gn;
    const auto v4 = m_gn1 - m_gn;

    const auto miu_prev = m_M(0, 0);
    const auto miu_next1 = 1.0 / (v1.dot(e) / v1.dot(v1) + 1.0 / miu_prev);
    const auto miu_next2 = 1.0 / (v2.dot(e) / v1.dot(v2) + 1.0 / miu_prev);
    const auto miu_next3 = 1.0 / (v3.dot(e) / v1.dot(v3) + 1.0 / miu_prev);
    const auto miu_next4 = 1.0 / (v4.dot(e) / v1.dot(v4) + 1.0 / miu_prev);

    const auto miu_next = std::min({std::isfinite(miu_next1) ? miu_next1 : std::numeric_limits<scalar_t>::max(),
                                    std::isfinite(miu_next2) ? miu_next2 : std::numeric_limits<scalar_t>::max(),
                                    std::isfinite(miu_next3) ? miu_next3 : std::numeric_limits<scalar_t>::max(),
                                    std::isfinite(miu_next4) ? miu_next4 : std::numeric_limits<scalar_t>::max()});

    if (miu_next != std::numeric_limits<scalar_t>::max())
    {
        m_M.diagonal().array() = miu_next;
    }
}

void quasi_t::update_sr1()
{
    const auto r = 1e-8;
    const auto e = m_xn1 - m_xn;
    const auto v = m_gn1 - m_gn;

    // TODO: which v should use?!
    // TODO: safeguard SR1 like in quasi-newton methods?!

    const auto Me = m_M * e;
    const auto should_apply = std::fabs(e.dot(Me + v)) >= r * e.norm() * (Me + v).norm();

    if (should_apply)
    {
        m_M -= (Me * Me.transpose()) / e.dot(Me + v);
    }
}

void quasi_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(parameter_t::make_enum(scat(prefix, "::prox::type"), quasi_type::sr1));
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::prox::tau_min"), 0.0, LT, 1e-5, LT, 1e+9));
}

quasi_t quasi_t::make(const solver_state_t& state, const configurable_t& c, const string_t& prefix)
{
    const auto type = c.parameter(scat(prefix, "::prox::type")).value<quasi_type>();
    const auto tau_min  = c.parameter(scat(prefix, "::prox::tau_min")).value<scalar_t>();

    return {state, type, tau_min};
}
