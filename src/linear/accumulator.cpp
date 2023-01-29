#include <nano/linear/accumulator.h>

using namespace nano::linear;

accumulator_t::accumulator_t() = default;

accumulator_t::accumulator_t(const tensor_size_t isize, const tensor_size_t tsize, const bool g1, const bool g2)
{
    if (g1)
    {
        m_gb1.resize(tsize);
        m_gW1.resize(tsize, isize);
        if (g2)
        {
            m_gb2.resize(tsize);
            m_gW2.resize(tsize, isize);
        }
    }

    clear();
}

void accumulator_t::clear()
{
    m_vm1 = 0.0;
    m_vm2 = 0.0;
    m_gb1.zero();
    m_gb2.zero();
    m_gW1.zero();
    m_gW2.zero();
}

accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
{
    m_vm1 += other.m_vm1;
    m_vm2 += other.m_vm2;
    m_gb1.vector() += other.m_gb1.vector();
    m_gW1.vector() += other.m_gW1.vector();
    m_gb2.vector() += other.m_gb2.vector();
    m_gW2.vector() += other.m_gW2.vector();
    return *this;
}

accumulator_t& accumulator_t::operator/=(const tensor_size_t samples)
{
    m_vm1 /= static_cast<scalar_t>(samples);
    m_vm2 /= static_cast<scalar_t>(samples);
    m_gb1.vector() /= static_cast<scalar_t>(samples);
    m_gW1.vector() /= static_cast<scalar_t>(samples);
    m_gb2.vector() /= static_cast<scalar_t>(samples);
    m_gW2.vector() /= static_cast<scalar_t>(samples);
    return *this;
}
