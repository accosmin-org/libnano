#pragma once

#include <nano/datasource.h>

namespace nano
{
///
/// \brief synthetic dataset:
///     the targets is a random affine transformation of the flatten input features.
///
/// NB: uniformly-distributed noise is added to targets if noise() > 0 (e.g. to evaluate robustness to noise).
/// NB: only %modulo features are taken into account (e.g. to test and evaluate feature selection).
///
class NANO_PUBLIC linear_datasource_t final : public datasource_t
{
public:
    ///
    /// \brief constructor
    ///
    linear_datasource_t();

    ///
    /// \brief @see clonable_t
    ///
    rdatasource_t clone() const override;

    ///
    /// \brief return the ground truth bias used to generate the synthetic dataset.
    ///
    const tensor1d_t& bias() const { return m_bias; }

    ///
    /// \brief return the ground truth weights used to generate the synthetic dataset.
    ///
    const tensor2d_t& weights() const { return m_weights; }

private:
    template <template <typename, size_t> class tstorage, typename tscalar, size_t trank, typename thitter>
    void setter(const tensor_size_t feature, const tensor_t<tstorage, tscalar, trank>& fvalues, const thitter& hitter)
    {
        for (tensor_size_t sample = 0, samples = fvalues.template size<0>(); sample < samples; ++sample)
        {
            if (hitter())
            {
                if constexpr (trank == 1)
                {
                    set(sample, feature, fvalues(sample));
                }
                else
                {
                    set(sample, feature, fvalues.tensor(sample));
                }
            }
        }
    }

    ///
    /// \brief @see datasource_t
    ///
    void do_load() override;

    // attributes
    tensor1d_t m_bias;    ///< 1D bias vector that offsets the output
    tensor2d_t m_weights; ///< 2D weight matrix that maps the input to the output
};
} // namespace nano
