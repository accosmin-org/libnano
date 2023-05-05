#include <nano/core/sampling.h>
#include <nano/gboost/sampler.h>

using namespace nano;
using namespace nano::gboost;

sampler_t::sampler_t(const indices_t& samples, const subsample_type type, const uint64_t seed)
    : m_samples(samples)
    , m_type(type)
    , m_rng(make_rng(seed))
    , m_weights((m_type == subsample_type::off || m_type == subsample_type::bootstrap) ? tensor_size_t{0}
                                                                                       : samples.size())
{
}

indices_t sampler_t::sample(const tensor2d_t& errors_losses, const tensor4d_t& gradients)
{
    switch (m_type)
    {
    case subsample_type::bootstrap:
    {
        return sample_with_replacement(m_samples, m_samples.size(), m_rng);
    }

    case subsample_type::wei_loss_bootstrap:
    {
        for (tensor_size_t i = 0, size = m_samples.size(); i < size; ++i)
        {
            m_weights(i) = errors_losses(1, m_samples(i)); // NB: loss value as sample weight!
        }
        return sample_with_replacement(m_samples, m_weights, m_weights.size(), m_rng);
    }

    case subsample_type::wei_grad_bootstrap:
    {
        for (tensor_size_t i = 0, size = m_samples.size(); i < size; ++i)
        {
            m_weights(i) = gradients.vector(m_samples(i)).lpNorm<2>(); // NB: gradient magnitude as sample weight!
        }
        return sample_with_replacement(m_samples, m_weights, m_weights.size(), m_rng);
    }

    case subsample_type::off:
    {
        return m_samples;
    }

    default:
    {
        assert(false);
        return indices_t{};
    }
    }
}
