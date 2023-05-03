#include <nano/gboost/accumulator.h>

using namespace nano;
using namespace nano::gboost;

accumulator_t::accumulator_t(const tensor_size_t tsize)
    : m_gb1(vector_t::Zero(tsize))
{
}

void accumulator_t::clear()
{
    m_vm1         = 0.0;
    m_gb1.array() = 0.0;
}

accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
{
    m_vm1 += other.m_vm1;
    m_gb1 += other.m_gb1;
    return *this;
}

accumulator_t& accumulator_t::operator/=(const tensor_size_t samples)
{
    m_vm1 /= static_cast<scalar_t>(samples);
    m_gb1 /= static_cast<scalar_t>(samples);
    return *this;
}

void accumulator_t::update(const tensor1d_cmap_t& values)
{
    m_vm1 += values.array().sum();
}

scalar_t accumulator_t::vgrad(vector_t* gx) const
{
    if (gx != nullptr)
    {
        *gx = m_gb1;
    }
    return m_vm1;
}
