#include <nano/linear/accumulator.h>

using namespace nano::linear;

accumulator_t::accumulator_t() = default;

accumulator_t::accumulator_t(const tensor_size_t isize, const tensor_size_t tsize)
    : m_gb(tsize)
    , m_gw(tsize, isize)
    , m_hx(tsize * isize + tsize, tsize * isize + tsize)
{
    clear();
}

void accumulator_t::clear()
{
    m_fx = 0.0;
    m_gb.zero();
    m_gw.zero();
    m_hx.zero();
}

accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
{
    m_fx += other.m_fx;
    m_gb += other.m_gb;
    m_gw += other.m_gw;
    m_hx += other.m_hx;
    return *this;
}

accumulator_t& accumulator_t::operator/=(const tensor_size_t samples)
{
    m_fx /= static_cast<scalar_t>(samples);
    m_gb /= static_cast<scalar_t>(samples);
    m_gw /= static_cast<scalar_t>(samples);
    m_hx /= static_cast<scalar_t>(samples);
    return *this;
}
