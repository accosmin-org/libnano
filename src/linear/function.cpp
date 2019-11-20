#include <nano/numeric.h>
#include <nano/linear/util.h>
#include <nano/linear/function.h>

using namespace nano;

template <typename ttensor>
static void zero(std::vector<ttensor>& tensors)
{
    for (size_t i = 0; i < tensors.size(); ++ i)
    {
        tensors[i].zero();
    }
}

template <typename ttensor>
static const auto& acc0(std::vector<ttensor>& tensors, const tensor_size_t samples)
{
    for (size_t i = 1; i < tensors.size(); ++ i)
    {
        tensors[0].vector() += tensors[i].vector();
    }
    tensors[0].vector() /= static_cast<scalar_t>(samples);
    return tensors[0];
}

linear_function_t::linear_function_t(const loss_t& loss, const iterator_t& iterator, const fold_t fold) :
    function_t("linear", (::nano::size(iterator.idim()) + 1) * ::nano::size(iterator.tdim()), convexity::yes),
    m_loss(loss),
    m_iterator(iterator),
    m_fold(fold),
    m_isize(::nano::size(iterator.idim())),
    m_tsize(::nano::size(iterator.tdim())),
    m_values(m_iterator.samples(m_fold)),
    m_vgrads(tpool_t::size()),
    m_gb1s(tpool_t::size(), tensor1d_t{m_tsize}),
    m_gb2s(tpool_t::size(), tensor1d_t{m_tsize}),
    m_gW1s(tpool_t::size(), tensor2d_t{m_isize, m_tsize}),
    m_gW2s(tpool_t::size(), tensor2d_t{m_isize, m_tsize})
{
    assert(m_isize > 0);
    assert(m_tsize > 0);
}

scalar_t linear_function_t::vgrad(const vector_t& x, vector_t* gx) const
{
    assert(!gx || gx->size() == x.size());
    assert(x.size() == (m_isize + 1) * m_tsize);

    const auto b = bias(x);
    const auto W = weights(x);

    zero(m_gb1s);
    zero(m_gW1s);
    zero(m_gb2s);
    zero(m_gW2s);

    m_iterator.shuffle(m_fold);

    ::nano::linear::iterate(m_iterator, m_fold, batch(), W, b, [&] (
        const tensor4d_t& inputs, const tensor4d_t& targets, const tensor4d_t& outputs,
        const tensor_size_t begin, const tensor_size_t end, const size_t tnum)
    {
        m_loss.value(targets, outputs, m_values.slice(begin, end - begin));

        if (gx != nullptr)
        {
            m_loss.vgrad(targets, outputs, m_vgrads[tnum]);

            const auto imatrix = inputs.reshape(end - begin, W.rows()).matrix();
            const auto gmatrix = m_vgrads[tnum].reshape(end - begin, W.cols()).matrix();

            m_gb1s[tnum].vector() += gmatrix.colwise().sum();
            m_gW1s[tnum].matrix() += imatrix.transpose() * gmatrix;

            if (vAreg() > 0)
            {
                const auto vvector = m_values.slice(begin, end - begin).vector();

                m_gb2s[tnum].vector() += gmatrix.transpose() * vvector;
                m_gW2s[tnum].matrix() += (imatrix.array().colwise() * vvector.array()).matrix().transpose() * gmatrix;
            }
        }
    });

    const scalar_t values_mean = m_values.array().mean();

    // OK, normalize and add the regularization terms
    if (gx != nullptr)
    {
        const auto& gb1 = acc0(m_gb1s, m_values.size());
        const auto& gW1 = acc0(m_gW1s, m_values.size());

        bias(*gx) = gb1;
        weights(*gx) = gW1;

        // NB: the variance-based regularization term must be the first!
        if (vAreg() > 0)
        {
            const auto& gb2 = acc0(m_gb2s, m_values.size());
            const auto& gW2 = acc0(m_gW2s, m_values.size());

            bias(*gx).array() += vAreg() * (gb2.array() - values_mean * bias(*gx).array()) * 2;
            weights(*gx).array() += vAreg() * (gW2.array() - values_mean * weights(*gx).array()) * 2;
        }
        if (l1reg() > 0)
        {
            weights(*gx).array() += l1reg() * W.array().sign() / W.size();
        }
        if (l2reg() > 0)
        {
            weights(*gx).array() += l2reg() * W.array() * 2 / W.size();
        }
    }

    return  values_mean +
            ((l1reg() > 0) ? (l1reg() * W.array().abs().mean()) : scalar_t(0)) +
            ((l2reg() > 0) ? (l2reg() * W.array().square().mean()) : scalar_t(0)) +
            ((vAreg() > 0) ? (vAreg() * (m_values.array().square().mean() - values_mean * values_mean)) : scalar_t(0));
}
