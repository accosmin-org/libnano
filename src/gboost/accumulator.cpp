#include <nano/gboost/accumulator.h>

using namespace nano;
using namespace nano::gboost;

accumulator_t::accumulator_t(const tensor_size_t tsize)
    : m_gx(vector_t::zero(tsize))
    , m_hx(matrix_t::zero(tsize, tsize))
{
}

void accumulator_t::clear()
{
    m_fx = 0.0;
    m_gx.zero();
    m_hx.zero();
}

accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
{
    m_fx += other.m_fx;
    m_gx += other.m_gx;
    m_hx += other.m_hx;
    return *this;
}

accumulator_t& accumulator_t::operator/=(const tensor_size_t samples)
{
    m_fx /= static_cast<scalar_t>(samples);
    m_gx /= static_cast<scalar_t>(samples);
    m_hx /= static_cast<scalar_t>(samples);
    return *this;
}

scalar_t accumulator_t::value(vector_map_t gx, matrix_map_t hx) const
{
    if (gx.dims() == m_gx.dims())
    {
        gx = m_gx;
    }
    if (hx.dims() == m_hx.dims())
    {
        hx = m_hx;
    }
    return m_fx;
}
