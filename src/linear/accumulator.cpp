#include <nano/linear/accumulator.h>

using namespace nano::linear;

accumulator_t::accumulator_t() = default;

accumulator_t::accumulator_t(const tensor_size_t isize, const tensor_size_t tsize)
    : m_gb1(tsize)
    , m_gW1(tsize, isize)
    , m_HbW(tsize * isize + tsize, tsize * isize + tsize)
{
    clear();
}

void accumulator_t::clear()
{
    m_vm1 = 0.0;
    m_gb1.zero();
    m_gW1.zero();
    m_HbW.zero();
}

accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
{
    m_vm1 += other.m_vm1;
    m_gb1 += other.m_gb1;
    m_gW1 += other.m_gW1;
    m_HbW += other.m_HbW;
    return *this;
}

accumulator_t& accumulator_t::operator/=(const tensor_size_t samples)
{
    m_vm1 /= static_cast<scalar_t>(samples);
    m_gb1 /= static_cast<scalar_t>(samples);
    m_gW1 /= static_cast<scalar_t>(samples);
    m_HbW /= static_cast<scalar_t>(samples);
    return *this;
}
