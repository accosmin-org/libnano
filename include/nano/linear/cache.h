#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

namespace nano::linear
{
    ///
    /// \brief cumulates partial results per thread useful in evaluating the linear functions.
    ///
    class NANO_PUBLIC cache_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        cache_t();

        ///
        /// \brief constructor
        ///
        cache_t(tensor_size_t isize, tensor_size_t tsize, bool g1, bool g2);

        ///
        /// \brief reset accumulators.
        ///
        void clear();

        ///
        /// \brief cumulate partial results.
        ///
        cache_t& operator+=(const cache_t& other);

        ///
        /// \brief normalize the cumulated results with the given number of samples
        ///
        cache_t& operator/=(tensor_size_t samples);

        ///
        /// \brief map-reduce the cumulated partial results over the given threads
        /// NB: the first thread's cache is used to cumulate the cache from all the other threads.
        ///
        static const cache_t& reduce(std::vector<cache_t>& caches, tensor_size_t samples);

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
