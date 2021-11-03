#include <nano/linear/cache.h>

using namespace nano::linear;

cache_t::cache_t() = default;

cache_t::cache_t(tensor_size_t isize, tensor_size_t tsize, bool g1, bool g2)
{
    if (g1)
    {
        m_gb1.resize(tsize);
        m_gW1.resize(isize, tsize);
        if (g2)
        {
            m_gb2.resize(tsize);
            m_gW2.resize(isize, tsize);
        }
    }

    m_gb1.zero();
    m_gb2.zero();
    m_gW1.zero();
    m_gW2.zero();
}

cache_t& cache_t::operator+=(const cache_t& other)
{
    m_vm1 += other.m_vm1;
    m_vm2 += other.m_vm2;
    m_gb1.vector() += other.m_gb1.vector();
    m_gW1.vector() += other.m_gW1.vector();
    m_gb2.vector() += other.m_gb2.vector();
    m_gW2.vector() += other.m_gW2.vector();
    return *this;
}

cache_t& cache_t::operator/=(tensor_size_t samples)
{
    m_vm1 /= static_cast<scalar_t>(samples);
    m_vm2 /= static_cast<scalar_t>(samples);
    m_gb1.vector() /= static_cast<scalar_t>(samples);
    m_gW1.vector() /= static_cast<scalar_t>(samples);
    m_gb2.vector() /= static_cast<scalar_t>(samples);
    m_gW2.vector() /= static_cast<scalar_t>(samples);
    return *this;
}

const cache_t& cache_t::reduce(std::vector<cache_t>& caches, tensor_size_t samples)
{
    auto& cache0 = caches[0];
    for (size_t i = 1; i < caches.size(); ++ i)
    {
        cache0 += caches[i];
    }
    return (cache0 /= samples);
}
