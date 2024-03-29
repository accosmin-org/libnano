#pragma once

#include <nano/core/random.h>
#include <nano/gboost/enums.h>
#include <nano/tensor.h>

namespace nano::gboost
{
///
/// \brief utility to select samples for fitting weak learners.
///
class NANO_PUBLIC sampler_t
{
public:
    ///
    /// \brief constructor
    ///
    sampler_t(const indices_t& samples, gboost_subsample, uint64_t seed, scalar_t ratio);

    ///
    /// \brief returns the samples to use for fitting weak learners.
    ///
    indices_t sample(const tensor2d_t& errors_losses, const tensor4d_t& gradients);

private:
    // attributes
    const indices_t& m_samples;                     ///< training samples to select from
    gboost_subsample m_type{gboost_subsample::off}; ///<
    rng_t            m_rng;                         ///<
    scalar_t         m_ratio{1.0};                  ///<
    tensor1d_t       m_weights;                     ///< per-sample weight
};
} // namespace nano::gboost
