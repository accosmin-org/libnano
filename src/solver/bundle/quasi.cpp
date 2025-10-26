#include <solver/bundle/quasi.h>

using namespace nano;
using namespace nano::bundle;

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

quasi_t::quasi_t(const tensor_size_t dims, const quasi_type type)
    : m_M(matrix_t::identity(dims, dims))
    , m_xn(vector_t::zero(dims))
    , m_xn1(vector_t::zero(dims))
    , m_gn(vector_t::zero(dims))
    , m_gn1(vector_t::zero(dims))
    , m_Gn(vector_t::zero(dims))
    , m_Gn1(vector_t::zero(dims))
    , m_type(type)
{
}

quasi_t::quasi_t(const solver_state_t& state, const quasi_type type)
    : quasi_t(state.x().size(), type)
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
        const auto e = m_xn1 - m_xn;
        const auto v = m_Gn1 - m_Gn;

        // TODO: implement the SR1 version

        const auto miu         = m_M(0, 0);
        m_M.diagonal().array() = 1.0 / (v.dot(e) / v.dot(v) + 1.0 / miu);
    }

    return m_M;
}

void quasi_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(parameter_t::make_enum(scat(prefix, "::quasi::type"), quasi_type::miu));
}

quasi_t quasi_t::make(const solver_state_t& state, const configurable_t& c, const string_t& prefix)
{
    const auto type = c.parameter(scat(prefix, "::quasi::type")).value<quasi_type>();

    return {state, type};
}
