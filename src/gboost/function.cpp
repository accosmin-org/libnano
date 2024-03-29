#include <nano/core/reduce.h>
#include <nano/dataset.h>
#include <nano/gboost/function.h>

using namespace nano;
using namespace nano::gboost;

namespace
{
void clear(accumulators_t& accumulators)
{
    for (auto& accumulator : accumulators)
    {
        accumulator.clear();
    }
}
} // namespace

scale_function_t::scale_function_t(const targets_iterator_t& iterator, const loss_t& loss, const cluster_t& cluster,
                                   const tensor4d_t& soutputs, const tensor4d_t& woutputs)
    : function_t("gboost-scale", cluster.groups())
    , m_iterator(iterator)
    , m_loss(loss)
    , m_cluster(cluster)
    , m_soutputs(soutputs)
    , m_woutputs(woutputs)
    , m_values(m_iterator.samples().size())
    , m_vgrads(cat_dims(m_iterator.samples().size(), m_iterator.dataset().target_dims()))
    , m_outputs(cat_dims(m_iterator.samples().size(), m_iterator.dataset().target_dims()))
    , m_accumulators(m_iterator.concurrency(), accumulator_t{this->size()})
{
    assert(m_soutputs.dims() == m_woutputs.dims());
    assert(m_soutputs.dims() == cat_dims(iterator.dataset().samples(), iterator.dataset().target_dims()));

    smooth(loss.smooth() ? smoothness::yes : smoothness::no);
    convex(loss.convex() ? convexity::yes : convexity::no);
}

rfunction_t scale_function_t::clone() const
{
    return std::make_unique<scale_function_t>(*this);
}

scalar_t scale_function_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    assert(gx.size() == 0 || gx.size() == x.size());
    assert(x.size() == m_cluster.groups());

    ::clear(m_accumulators);

    const auto& samples = m_iterator.samples();

    m_iterator.loop(
        [&](const tensor_range_t& range, const size_t tnum, const tensor4d_cmap_t& targets)
        {
            assert(tnum < m_accumulators.size());
            auto& accumulator = m_accumulators[tnum];

            const auto begin = range.begin();
            const auto end   = range.end();

            // output = output(strong learner) + scale * output(weak learner)
            auto outputs = m_outputs.slice(range);
            for (tensor_size_t i = begin; i < end; ++i)
            {
                const auto group          = m_cluster.group(samples(i));
                const auto scale          = (group < 0) ? 0.0 : x(group);
                outputs.vector(i - begin) = m_soutputs.vector(samples(i)) + scale * m_woutputs.vector(samples(i));
            }

            auto values = m_values.slice(range);
            m_loss.value(targets, outputs, values);
            accumulator.update(values);

            if (gx.size() == x.size())
            {
                auto vgrads = m_vgrads.slice(range);
                m_loss.vgrad(targets, outputs, vgrads);

                for (tensor_size_t i = begin; i < end; ++i)
                {
                    const auto group = m_cluster.group(samples(i));
                    if (group < 0)
                    {
                        continue;
                    }
                    const auto gw = vgrads.vector(i - begin).dot(m_woutputs.vector(samples(i)));

                    accumulator.m_gb1(group) += gw;
                }
            }
        });

    // OK
    const auto& accumulator = ::nano::sum_reduce(m_accumulators, samples.size());
    return accumulator.vgrad(gx);
}

bias_function_t::bias_function_t(const targets_iterator_t& iterator, const loss_t& loss)
    : function_t("gboost-bias", ::nano::size(iterator.dataset().target_dims()))
    , m_iterator(iterator)
    , m_loss(loss)
    , m_values(m_iterator.samples().size())
    , m_vgrads(cat_dims(m_iterator.samples().size(), m_iterator.dataset().target_dims()))
    , m_outputs(cat_dims(m_iterator.samples().size(), m_iterator.dataset().target_dims()))
    , m_accumulators(m_iterator.concurrency(), accumulator_t{this->size()})
{
    smooth(loss.smooth() ? smoothness::yes : smoothness::no);
    convex(loss.convex() ? convexity::yes : convexity::no);
}

rfunction_t bias_function_t::clone() const
{
    return std::make_unique<bias_function_t>(*this);
}

scalar_t bias_function_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto& samples = m_iterator.samples();
    const auto  tsize   = nano::size(m_iterator.dataset().target_dims());

    assert(gx.size() == 0 || gx.size() == x.size());
    assert(x.size() == tsize);

    ::clear(m_accumulators);

    m_iterator.loop(
        [&](const tensor_range_t& range, const size_t tnum, const tensor4d_cmap_t& targets)
        {
            assert(tnum < m_accumulators.size());
            auto& accumulator = m_accumulators[tnum];

            // output = bias (fixed vector)
            auto outputs                                         = m_outputs.slice(range);
            outputs.reshape(range.size(), -1).matrix().rowwise() = x.transpose();

            auto values = m_values.slice(range);
            m_loss.value(targets, outputs, values);
            accumulator.update(values);

            if (gx.size() > 0)
            {
                auto vgrads = m_vgrads.slice(range);
                m_loss.vgrad(targets, outputs, vgrads);

                accumulator.m_gb1 += vgrads.reshape(range.size(), tsize).matrix().colwise().sum();
            }
        });

    // OK
    const auto& accumulator = ::nano::sum_reduce(m_accumulators, samples.size());
    return accumulator.vgrad(gx);
}

grads_function_t::grads_function_t(const targets_iterator_t& iterator, const loss_t& loss)
    : function_t("gboost-grads", iterator.samples().size() * nano::size(iterator.dataset().target_dims()))
    , m_iterator(iterator)
    , m_loss(loss)
    , m_values(iterator.samples().size())
    , m_vgrads(cat_dims(iterator.samples().size(), iterator.dataset().target_dims()))
{
    smooth(loss.smooth() ? smoothness::yes : smoothness::no);
    convex(loss.convex() ? convexity::yes : convexity::no);
}

rfunction_t grads_function_t::clone() const
{
    return std::make_unique<grads_function_t>(*this);
}

scalar_t grads_function_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto& samples = m_iterator.samples();
    const auto  odims   = cat_dims(samples.size(), m_iterator.dataset().target_dims());

    assert(gx.size() == 0 || gx.size() == x.size());
    assert(x.size() == nano::size(odims));

    const auto& grads = gradients(map_tensor(x.data(), odims));
    if (gx.size() > 0)
    {
        gx = grads.vector() / static_cast<scalar_t>(samples.size());
    }

    // OK
    return m_values.vector().mean();
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

    return m_vgrads;
}
