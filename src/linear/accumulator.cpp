#include <nano/linear/accumulator.h>

using namespace nano::linear;

accumulator_t::accumulator_t() = default;

accumulator_t::accumulator_t(const tensor_size_t isize, const tensor_size_t tsize)
{
    m_gb1.resize(tsize);
    m_gW1.resize(tsize, isize);

    clear();
}

void accumulator_t::clear()
{
    m_vm1 = 0.0;
    m_gb1.zero();
    m_gW1.zero();
}

accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
{
    m_vm1 += other.m_vm1;
    m_gb1 += other.m_gb1;
    m_gW1 += other.m_gW1;
    return *this;
}

accumulator_t& accumulator_t::operator/=(const tensor_size_t samples)
{
    m_vm1 /= static_cast<scalar_t>(samples);
    m_gb1 /= static_cast<scalar_t>(samples);
    m_gW1 /= static_cast<scalar_t>(samples);
    return *this;
}
