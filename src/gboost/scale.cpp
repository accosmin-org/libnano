#include <nano/gboost/util.h>
#include <nano/gboost/scale.h>

using namespace nano;

class cache_t
{
public:

    explicit cache_t(const tensor_size_t tsize = 1) :
        m_gb1(vector_t::Zero(tsize)),
        m_gb2(vector_t::Zero(tsize))
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

    // attributes
    scalar_t    m_vm1{0}, m_vm2{0}; ///< first and second order momentum of the loss values
    vector_t    m_gb1{0}, m_gb2{0}; ///< first and second order momentum of the gradient wrt scale
};

gboost_scale_function_t::gboost_scale_function_t(const loss_t& loss, const dataset_t& dataset, fold_t fold,
    const cluster_t& cluster, const tensor4d_t& outputs, const tensor4d_t& woutputs) :
    function_t("gboost_scale", cluster.groups(), convexity::yes),
    m_loss(loss),
    m_dataset(dataset),
    m_fold(fold),
    m_cluster(cluster),
    m_outputs(outputs),
    m_woutputs(woutputs)
{
    assert(m_outputs.dims() == m_woutputs.dims());
    assert(m_outputs.dims() == cat_dims(m_dataset.samples(m_fold), m_dataset.tdim()));
}

scalar_t gboost_scale_function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    assert(!gx || gx->size() == x.size());
    assert(x.size() == m_cluster.groups());

    std::vector<cache_t> caches(tpool_t::size(), cache_t{x.size()});

    m_dataset.loop(execution::par, m_fold, batch(), [&] (tensor_range_t range, size_t tnum)
    {
        assert(tnum < caches.size());
        auto& cache = caches[tnum];

        const auto targets = m_dataset.targets(m_fold, range);

        // output = output(strong learner) + scale * output(weak learner)
        tensor4d_t outputs(targets.dims());
        for (tensor_size_t i = range.begin(); i < range.end(); ++ i)
        {
            const auto group = m_cluster.group(i);
            const auto scale = (group < 0) ? 0.0 : x(group);
            outputs.vector(i - range.begin()) = m_outputs.vector(i) + scale * m_woutputs.vector(i);
        }

        tensor1d_t values;
        m_loss.value(targets, outputs, values);

        cache.m_vm1 += values.array().sum();
        if (vAreg() > 0)
        {
            cache.m_vm2 += values.array().square().sum();
        }

        if (gx != nullptr)
        {
            tensor4d_t vgrads;
            m_loss.vgrad(targets, outputs, vgrads);

            for (tensor_size_t i = range.begin(); i < range.end(); ++ i)
            {
                const auto group = m_cluster.group(i);
                if (group < 0)
                {
                    continue;
                }
                const auto gw = vgrads.vector(i - range.begin()).dot(m_woutputs.vector(i));

                cache.m_gb1(group) += gw;
                if (vAreg() > 0)
                {
                    cache.m_gb2(group) += gw * values(i - range.begin());
                }
            }
        }
    });

    const auto& cache0 = ::nano::gboost::sum_reduce(caches, m_dataset.samples(m_fold));

    // OK, normalize and add the regularization terms
    if (gx != nullptr)
    {
        *gx = cache0.m_gb1;
        if (vAreg() > 0)
        {
            *gx += vAreg() * (cache0.m_gb2 - cache0.m_vm1 * cache0.m_gb1) * 2;
        }
    }

    return  cache0.m_vm1 +
            ((vAreg() > 0) ? (vAreg() * (cache0.m_vm2 - cache0.m_vm1 * cache0.m_vm1)) : scalar_t(0));
}
