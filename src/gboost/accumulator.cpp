#include <nano/gboost/accumulator.h>

using namespace nano;
using namespace nano::gboost;

accumulator_t::accumulator_t(const tensor_size_t tsize)
    : m_gb1(vector_t::zero(tsize))
    , m_hb2(matrix_t::zero(tsize, tsize))
{
}

void accumulator_t::clear()
{
    m_vm1         = 0.0;
    m_gb1.array() = 0.0;
    m_hb2.array() = 0.0;
}

accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
{
    m_vm1 += other.m_vm1;
    m_gb1 += other.m_gb1;
    m_hb2 += other.m_hb2;
    return *this;
}

accumulator_t& accumulator_t::operator/=(const tensor_size_t samples)
{
    m_vm1 /= static_cast<scalar_t>(samples);
    m_gb1 /= static_cast<scalar_t>(samples);
    m_hb2 /= static_cast<scalar_t>(samples);
    return *this;
}

void accumulator_t::update(const tensor1d_cmap_t& values)
{
    m_vm1 += values.array().sum();
}

scalar_t accumulator_t::vgrad(vector_map_t gx, matrix_map_t Hx) const
{
    if (gx.size() > 0)
    {
        gx = m_gb1;
    }
    if (Hx.size() > 0)
    {
        Hx = m_hb2;
    }
    return m_vm1;
}
