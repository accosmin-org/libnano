#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

namespace nano::linear
{
    ///
    /// \brief cumulates partial results per thread useful in evaluating the linear functions.
    ///
    class NANO_PUBLIC accumulator_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        accumulator_t();

        ///
        /// \brief constructor
        ///
        accumulator_t(tensor_size_t isize, tensor_size_t tsize, bool g1, bool g2);

        ///
        /// \brief reset the accumulator.
        ///
        void clear();

        ///
        /// \brief cumulate partial results.
        ///
        accumulator_t& operator+=(const accumulator_t& other);

        ///
        /// \brief normalize the cumulated results with the given number of samples.
        ///
        accumulator_t& operator/=(tensor_size_t samples);

        // attributes
        tensor4d_t m_outputs; ///< buffer: predictions
        tensor4d_t m_vgrads;  ///< buffer: gradients wrt predictions
        tensor1d_t m_values;  ///< buffer: loss values
        scalar_t   m_vm1{0};  ///< first order momentum of the loss values
        scalar_t   m_vm2{0};  ///< second order momentum of the loss values
        tensor1d_t m_gb1;     ///< first order momentum of the gradient wrt bias
        tensor1d_t m_gb2;     ///< second order momentum of the gradient wrt bias
        tensor2d_t m_gW1;     ///< first order momentum of the gradient wrt weights
        tensor2d_t m_gW2;     ///< second order momentum of the gradient wrt weights
    };

    using accumulators_t = std::vector<accumulator_t>;
} // namespace nano::linear
