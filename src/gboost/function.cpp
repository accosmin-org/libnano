#include <nano/core/reduce.h>
#include <nano/dataset.h>
#include <nano/gboost/function.h>

using namespace nano;
using namespace nano::gboost;

grads_function_t::grads_function_t(const targets_iterator_t& iterator, const loss_t& loss)
    : function_t("gboost-grads", iterator.samples().size() * nano::size(iterator.dataset().target_dims()))
    , m_iterator(iterator)
    , m_loss(loss)
    , m_values(iterator.samples().size())
    , m_vgrads(cat_dims(iterator.samples().size(), iterator.dataset().target_dims()))
    , m_vhesss(loss_t::make_hess_dims(iterator.samples().size(), iterator.dataset().target_dims()))
{
    smooth(loss.smooth() ? smoothness::yes : smoothness::no);
    convex(loss.convex() ? convexity::yes : convexity::no);
}

rfunction_t grads_function_t::clone() const
{
    return std::make_unique<grads_function_t>(*this);
}

scalar_t grads_function_t::do_eval(eval_t eval) const
{
    const auto& samples = m_iterator.samples();
    const auto  tsize   = ::nano::size(m_iterator.dataset().target_dims());
    const auto  odims   = cat_dims(samples.size(), m_iterator.dataset().target_dims());
    const auto  denom   = static_cast<scalar_t>(samples.size());

    const auto& grads = gradients(map_tensor(eval.m_x.data(), odims));
    if (eval.has_grad())
    {
        eval.m_gx = grads.vector() / denom;
    }
    if (eval.has_hess())
    {
        eval.m_hx.full(0.0);

        for (tensor_size_t sample = 0; sample < samples.size(); ++sample)
        {
            eval.m_hx.matrix().block(sample * tsize, sample * tsize, tsize, tsize) =
                m_vhesss.tensor(sample).reshape(tsize, tsize).matrix() / denom;
        }
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
            m_loss.vhess(targets, outputs.slice(range), m_vhesss.slice(range));
        });

    return m_vgrads;
}

bias_function_t::bias_function_t(const targets_iterator_t& iterator, const loss_t& loss)
    : function_t("gboost-bias", ::nano::size(iterator.dataset().target_dims()))
    , m_iterator(iterator)
    , m_loss(loss)
    , m_accumulators(m_iterator.concurrency(), accumulator_t{this->size()})
{
    smooth(loss.smooth() ? smoothness::yes : smoothness::no);
    convex(loss.convex() ? convexity::yes : convexity::no);
}

rfunction_t bias_function_t::clone() const
{
    return std::make_unique<bias_function_t>(*this);
}

scalar_t bias_function_t::do_eval(eval_t eval) const
{
    const auto& samples = m_iterator.samples();
    const auto  tdims   = m_iterator.dataset().target_dims();
    const auto  tsize   = nano::size(tdims);

    std::for_each(m_accumulators.begin(), m_accumulators.end(), [&](auto& accumulator) { accumulator.clear(); });

    m_iterator.loop(
        [&](const tensor_range_t& range, const size_t tnum, const tensor4d_cmap_t& targets)
        {
            assert(tnum < m_accumulators.size());
            auto& accumulator = m_accumulators[tnum];

            // output = bias (fixed vector)
            accumulator.m_outputs.resize(cat_dims(range.size(), tdims));
            accumulator.m_outputs.reshape(range.size(), -1).matrix().rowwise() = eval.m_x.transpose();

            // cumulate values
            m_loss.value(targets, accumulator.m_outputs, accumulator.m_loss_fx);
            accumulator.m_fx += accumulator.m_loss_fx.sum();

            // cumulate gradients
            if (eval.has_grad())
            {
                m_loss.vgrad(targets, accumulator.m_outputs, accumulator.m_loss_gx);

                const auto gmatrix = accumulator.m_loss_gx.reshape(range.size(), tsize);
                accumulator.m_gx += gmatrix.matrix().colwise().sum().transpose();
            }

            // cumulate hessians
            if (eval.has_hess())
            {
                m_loss.vhess(targets, accumulator.m_outputs, accumulator.m_loss_hx);

                const auto hmatrix = accumulator.m_loss_hx.reshape(range.size(), tsize * tsize);
                accumulator.m_hx.array() += hmatrix.matrix().colwise().sum().array();
            }
        });

    // OK
    const auto& accumulator = ::nano::sum_reduce(m_accumulators, samples.size());
    return accumulator.value(eval.m_gx, eval.m_hx);
}

scale_function_t::scale_function_t(const targets_iterator_t& iterator, const loss_t& loss, const cluster_t& cluster,
                                   const tensor4d_t& soutputs, const tensor4d_t& woutputs)
    : function_t("gboost-scale", cluster.groups())
    , m_iterator(iterator)
    , m_loss(loss)
    , m_cluster(cluster)
    , m_soutputs(soutputs)
    , m_woutputs(woutputs)
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

scalar_t scale_function_t::do_eval(eval_t eval) const
{
    std::for_each(m_accumulators.begin(), m_accumulators.end(), [&](auto& accumulator) { accumulator.clear(); });

    const auto& samples = m_iterator.samples();
    const auto  tdims   = m_iterator.dataset().target_dims();
    const auto  tsize   = nano::size(tdims);

    m_iterator.loop(
        [&](const tensor_range_t& range, const size_t tnum, const tensor4d_cmap_t& targets)
        {
            assert(tnum < m_accumulators.size());
            auto& accumulator = m_accumulators[tnum];

            const auto begin = range.begin();
            const auto end   = range.end();

            // output = output(strong learner) + scale * output(weak learner)
            accumulator.m_outputs.resize(cat_dims(range.size(), tdims));
            for (tensor_size_t i = begin; i < end; ++i)
            {
                const auto group                        = m_cluster.group(samples(i));
                const auto scale                        = (group < 0) ? 0.0 : eval.m_x(group);
                const auto soutput                      = m_soutputs.vector(samples(i));
                const auto woutput                      = m_woutputs.vector(samples(i));
                accumulator.m_outputs.vector(i - begin) = soutput + scale * woutput;
            }

            // cumulate values
            m_loss.value(targets, accumulator.m_outputs, accumulator.m_loss_fx);
            accumulator.m_fx += accumulator.m_loss_fx.sum();

            // cumulate gradients
            if (eval.has_grad())
            {
                m_loss.vgrad(targets, accumulator.m_outputs, accumulator.m_loss_gx);

                for (tensor_size_t i = begin; i < end; ++i)
                {
                    const auto group = m_cluster.group(samples(i));
                    if (group < 0)
                    {
                        continue;
                    }

                    const auto woutput = m_woutputs.vector(samples(i));
                    const auto gvector = accumulator.m_loss_gx.vector(i - begin);
                    accumulator.m_gx(group) += gvector.dot(woutput);
                }
            }

            // cumulate hessians
            if (eval.has_hess())
            {
                m_loss.vhess(targets, accumulator.m_outputs, accumulator.m_loss_hx);

                for (tensor_size_t i = begin; i < end; ++i)
                {
                    const auto group = m_cluster.group(samples(i));
                    if (group < 0)
                    {
                        continue;
                    }

                    const auto woutput = m_woutputs.vector(samples(i));
                    const auto hmatrix = accumulator.m_loss_hx.tensor(i - begin).reshape(tsize, tsize).matrix();
                    accumulator.m_hx(group, group) += woutput.transpose() * hmatrix * woutput;
                }
            }
        });

    // OK
    const auto& accumulator = ::nano::sum_reduce(m_accumulators, samples.size());
    return accumulator.value(eval.m_gx, eval.m_hx);
}
