#pragma once

#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief cumulates partial results per thread useful in evaluating the linear functions.
    ///
    class linear_cache_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        linear_cache_t() = default;

        ///
        /// \brief constructor
        ///
        linear_cache_t(const tensor_size_t isize, const tensor_size_t tsize, const bool g1, const bool g2)
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

        ///
        /// \brief cumulate partial results
        ///
        linear_cache_t& operator+=(const linear_cache_t& other)
        {
            m_vm1 += other.m_vm1;
            m_vm2 += other.m_vm2;
            m_gb1.vector() += other.m_gb1.vector();
            m_gW1.vector() += other.m_gW1.vector();
            m_gb2.vector() += other.m_gb2.vector();
            m_gW2.vector() += other.m_gW2.vector();
            return *this;
        }

        ///
        /// \brief normalize the cumulated results with the given number of samples
        ///
        linear_cache_t& operator/=(const tensor_size_t samples)
        {
            m_vm1 /= static_cast<scalar_t>(samples);
            m_vm2 /= static_cast<scalar_t>(samples);
            m_gb1.vector() /= static_cast<scalar_t>(samples);
            m_gW1.vector() /= static_cast<scalar_t>(samples);
            m_gb2.vector() /= static_cast<scalar_t>(samples);
            m_gW2.vector() /= static_cast<scalar_t>(samples);
            return *this;
        }

        ///
        /// \brief map-reduce the cumulated partial results over the given threads
        /// NB: the first thread's cache is used to cumulate the cache from all the other threads.
        ///
        static const auto& reduce(std::vector<linear_cache_t>& caches, const tensor_size_t samples)
        {
            auto& cache0 = caches[0];
            for (size_t i = 1; i < caches.size(); ++ i)
            {
                cache0 += caches[i];
            }
            return (cache0 /= samples);
        }

        // attributes
        tensor4d_t  m_outputs;      ///< buffer: predictions
        tensor4d_t  m_vgrads;       ///< buffer: gradients wrt predictions
        tensor1d_t  m_values;       ///< buffer: loss values
        scalar_t    m_vm1{0};       ///< first order momentum of the loss values
        scalar_t    m_vm2{0};       ///< second order momentum of the loss values
        tensor1d_t  m_gb1, m_gb2;   ///< first and second order momentum of the gradient wrt bias
        tensor2d_t  m_gW1, m_gW2;   ///< first and second order momentum of the gradient wrt weights
    };
}
