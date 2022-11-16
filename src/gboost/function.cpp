#include <nano/dataset.h>
#include <nano/gboost/function.h>
#include <nano/wlearner/reduce.h>

using namespace nano;
using namespace nano::gboost;

class cache_t
{
public:
    explicit cache_t(const tensor_size_t tsize = 1)
        : m_gb1(vector_t::Zero(tsize))
        , m_gb2(vector_t::Zero(tsize))
    {
    }

    cache_t& operator+=(const cache_t& other)
    {
        m_vm1 += other.m_vm1;
        m_vm2 += other.m_vm2;
        m_gb1 += other.m_gb1;
        m_gb2 += other.m_gb2;
        return *this;
    }

    cache_t& operator/=(const tensor_size_t samples)
    {
        m_vm1 /= static_cast<scalar_t>(samples);
        m_vm2 /= static_cast<scalar_t>(samples);
        m_gb1 /= static_cast<scalar_t>(samples);
        m_gb2 /= static_cast<scalar_t>(samples);
        return *this;
    }

    void update(const tensor1d_t& values)
    {
        m_vm1 += values.array().sum();
        m_vm2 += values.array().square().sum();
    }

    auto vgrad(scalar_t vAreg, vector_t* gx) const
    {
        if (gx != nullptr)
        {
            *gx = m_gb1 + vAreg * (m_gb2 - m_vm1 * m_gb1) * 2;
        }
        return m_vm1 + vAreg * (m_vm2 - m_vm1 * m_vm1);
    }

    // attributes
    scalar_t m_vm1{0}, m_vm2{0}; ///< first and second order momentum of the loss values
    vector_t m_gb1{0}, m_gb2{0}; ///< first and second order momentum of the gradient wrt scale
};

scale_function_t::scale_function_t(const targets_iterator_t& iterator, const loss_t& loss, const scalar_t vAreg,
                                   const cluster_t& cluster, const tensor4d_t& outputs, const tensor4d_t& woutputs)
    : function_t("gboost-scale", cluster.groups())
    , m_iterator(iterator)
    , m_loss(loss)
    , m_vAreg(vAreg)
    , m_cluster(cluster)
    , m_outputs(outputs)
    , m_woutputs(woutputs)
{
    assert(m_outputs.dims() == m_woutputs.dims());
    assert(m_outputs.dims() == cat_dims(iterator.samples().size(), iterator.dataset().target_dims()));

    smooth(loss.smooth());
    convex(std::abs(vAreg) < std::numeric_limits<scalar_t>::epsilon() && loss.convex());
}

rfunction_t scale_function_t::clone() const
{
    return std::make_unique<scale_function_t>(*this);
}

scalar_t scale_function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    assert(!gx || gx->size() == x.size());
    assert(x.size() == m_cluster.groups());

    const auto& samples = m_iterator.samples();

    std::vector<cache_t> caches(m_iterator.concurrency(), cache_t{x.size()});
    m_iterator.loop(
        [&](const tensor_range_t& range, const size_t tnum, const tensor4d_cmap_t& targets)
        {
            assert(tnum < caches.size());
            auto& cache = caches[tnum];

            const auto begin = range.begin();
            const auto end   = range.end();

            // output = output(strong learner) + scale * output(weak learner)
            tensor4d_t outputs(targets.dims());
            for (tensor_size_t i = begin; i < end; ++i)
            {
                const auto group          = m_cluster.group(samples(i));
                const auto scale          = (group < 0) ? 0.0 : x(group);
                outputs.vector(i - begin) = m_outputs.vector(i) + scale * m_woutputs.vector(i);
            }

            tensor1d_t values;
            m_loss.value(targets, outputs, values);
            cache.update(values);

            if (gx != nullptr)
            {
                tensor4d_t vgrads;
                m_loss.vgrad(targets, outputs, vgrads);

                for (tensor_size_t i = begin; i < end; ++i)
                {
                    const auto group = m_cluster.group(samples(i));
                    if (group < 0)
                    {
                        continue;
                    }
                    const auto gw = vgrads.vector(i - begin).dot(m_woutputs.vector(i));

                    cache.m_gb1(group) += gw;
                    cache.m_gb2(group) += gw * values(i - begin);
                }
            }
        });

    // OK
    const auto& cache0 = ::nano::wlearner::sum_reduce(caches, samples.size());
    return cache0.vgrad(m_vAreg, gx);
}

bias_function_t::bias_function_t(const targets_iterator_t& iterator, const loss_t& loss, const scalar_t vAreg)
    : function_t("gboost-bias", ::nano::size(iterator.dataset().target_dims()))
    , m_iterator(iterator)
    , m_loss(loss)
    , m_vAreg(vAreg)
{
    smooth(loss.smooth());
    convex(std::abs(vAreg) < std::numeric_limits<scalar_t>::epsilon() && loss.convex());
}

rfunction_t bias_function_t::clone() const
{
    return std::make_unique<bias_function_t>(*this);
}

scalar_t bias_function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    const auto& samples = m_iterator.samples();
    const auto  tsize   = nano::size(m_iterator.dataset().target_dims());

    assert(!gx || gx->size() == x.size());
    assert(x.size() == tsize);

    std::vector<cache_t> caches(m_iterator.concurrency(), cache_t{x.size()});
    m_iterator.loop(
        [&](const tensor_range_t& range, const size_t tnum, const tensor4d_cmap_t& targets)
        {
            assert(tnum < caches.size());
            auto& cache = caches[tnum];

            // output = bias (fixed vector)
            tensor4d_t outputs(targets.dims());
            outputs.reshape(range.size(), -1).matrix().rowwise() = x.transpose();

            tensor1d_t values;
            m_loss.value(targets, outputs, values);
            cache.update(values);

            if (gx != nullptr)
            {
                tensor4d_t vgrads;
                m_loss.vgrad(targets, outputs, vgrads);
                const auto gmatrix = vgrads.reshape(range.size(), tsize).matrix();

                cache.m_gb1 += gmatrix.colwise().sum();
                cache.m_gb2 += gmatrix.transpose() * values.vector();
            }
        });

    // OK
    const auto& cache0 = ::nano::wlearner::sum_reduce(caches, samples.size());
    return cache0.vgrad(m_vAreg, gx);
}

grads_function_t::grads_function_t(const targets_iterator_t& iterator, const loss_t& loss, const scalar_t vAreg)
    : function_t("gboost-grads", iterator.samples().size() * nano::size(iterator.dataset().target_dims()))
    , m_iterator(iterator)
    , m_loss(loss)
    , m_vAreg(vAreg)
    , m_values(iterator.samples().size())
    , m_vgrads(cat_dims(iterator.samples().size(), iterator.dataset().target_dims()))
{
    smooth(loss.smooth());
    convex(std::abs(vAreg) < std::numeric_limits<scalar_t>::epsilon() && loss.convex());
}

rfunction_t grads_function_t::clone() const
{
    return std::make_unique<grads_function_t>(*this);
}

scalar_t grads_function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    const auto& samples = m_iterator.samples();
    const auto  odims   = cat_dims(samples.size(), m_iterator.dataset().target_dims());

    assert(!gx || gx->size() == x.size());
    assert(x.size() == nano::size(odims));

    const auto& grads = gradients(map_tensor(x.data(), odims));
    if (gx != nullptr)
    {
        *gx = grads.vector();
        *gx /= static_cast<scalar_t>(samples.size());
    }

    // OK
    const auto vm1 = m_values.vector().mean();
    const auto vm2 = m_values.array().square().mean();
    return vm1 + m_vAreg * (vm2 - vm1 * vm1);
}

const tensor4d_t& grads_function_t::gradients(const tensor4d_cmap_t& outputs) const
{
    assert(outputs.dims() == m_vgrads.dims());

    m_iterator.loop(
        [&](const tensor_range_t& range, size_t, const tensor4d_cmap_t& targets)
        {
            m_loss.value(targets, outputs.slice(range), m_values.slice(range));
            m_loss.vgrad(targets, outputs.slice(range), m_vgrads.slice(range));
        });

    const auto vm1 = m_values.vector().mean();
    // FIXME: rewrite this loop as a more efficient Eigen operation
    for (tensor_size_t i = 0; i < m_vgrads.size<0>(); ++i)
    {
        m_vgrads.vector(i) *= 1.0 + 2.0 * m_vAreg * (m_values(i) - vm1);
    }

    return m_vgrads;
}
