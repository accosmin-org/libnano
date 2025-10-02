#include <nano/linear/accumulator.h>

using namespace nano::linear;

accumulator_t::accumulator_t() = default;

accumulator_t::accumulator_t(const tensor_size_t isize, const tensor_size_t tsize)
    : m_gb(tsize)
    , m_gw(tsize, isize)
    , m_hww(tsize * isize, tsize * isize)
    , m_hwb(tsize * isize, tsize)
    , m_hbb(tsize, tsize)
{
    clear();
}

void accumulator_t::clear()
{
    m_fx = 0.0;
    m_gb.zero();
    m_gw.zero();
    m_hww.zero();
    m_hwb.zero();
    m_hbb.zero();
}

accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
{
    m_fx += other.m_fx;
    m_gb += other.m_gb;
    m_gw += other.m_gw;
    m_hww += other.m_hww;
    m_hwb += other.m_hwb;
    m_hbb += other.m_hbb;
    return *this;
}

accumulator_t& accumulator_t::operator/=(const tensor_size_t samples)
{
    m_fx /= static_cast<scalar_t>(samples);
    m_gb /= static_cast<scalar_t>(samples);
    m_gw /= static_cast<scalar_t>(samples);
    m_hww /= static_cast<scalar_t>(samples);
    m_hwb /= static_cast<scalar_t>(samples);
    m_hbb /= static_cast<scalar_t>(samples);
    return *this;
}
