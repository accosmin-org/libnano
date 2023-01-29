#include <nano/core/reduce.h>
#include <nano/dataset.h>
#include <nano/linear/function.h>
#include <nano/linear/util.h>

using namespace nano;

static auto isize(const flatten_iterator_t& iterator)
{
    return iterator.dataset().columns();
}

static auto tsize(const flatten_iterator_t& iterator)
{
    return ::nano::size(iterator.dataset().target_dims());
}

linear::function_t::function_t(const flatten_iterator_t& iterator, const loss_t& loss, scalar_t l1reg, scalar_t l2reg,
                               scalar_t vAreg)
    : ::nano::function_t("linear", (::isize(iterator) + 1) * ::tsize(iterator))
    , m_iterator(iterator)
    , m_loss(loss)
    , m_l1reg(l1reg)
    , m_l2reg(l2reg)
    , m_vAreg(vAreg)
    , m_isize(::isize(m_iterator))
    , m_tsize(::tsize(m_iterator))
    , m_accumulators(m_iterator.concurrency(), accumulator_t{m_isize, m_tsize, true, m_vAreg > 0.0})
{
    assert(m_isize > 0);
    assert(m_tsize > 0);

    convex(m_loss.convex() && m_vAreg <= 0.0);
    smooth(m_loss.smooth() && m_l1reg <= 0.0);
    strong_convexity(m_l2reg / static_cast<scalar_t>(m_isize * m_tsize));
}

rfunction_t linear::function_t::clone() const
{
    return std::make_unique<linear::function_t>(*this);
}

scalar_t linear::function_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    assert(!gx || gx->size() == x.size());
    assert(x.size() == (m_isize + 1) * m_tsize);

    const auto b = bias(x);
    const auto W = weights(x);

    for (auto& accumulator : m_accumulators)
    {
        accumulator.clear();
    }

    m_iterator.loop(
        [&](tensor_range_t range, size_t tnum, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
        {
            assert(tnum < m_accumulators.size());
            auto& accumulator = m_accumulators[tnum];

            ::nano::linear::predict(inputs, W, b, accumulator.m_outputs);
            m_loss.value(targets, accumulator.m_outputs, accumulator.m_values);

            const auto vvector = accumulator.m_values.vector();

            accumulator.m_vm1 += vvector.array().sum();
            if (m_vAreg > 0.0)
            {
                accumulator.m_vm2 += vvector.array().square().sum();
            }

            if (gx != nullptr)
            {
                m_loss.vgrad(targets, accumulator.m_outputs, accumulator.m_vgrads);

                const auto imatrix = inputs.matrix();
                const auto gmatrix = accumulator.m_vgrads.reshape(range.size(), m_tsize).matrix();

                accumulator.m_gb1.vector() += gmatrix.colwise().sum();
                accumulator.m_gW1.matrix() += gmatrix.transpose() * imatrix;

                if (m_vAreg > 0.0)
                {
                    accumulator.m_gb2.vector() += gmatrix.transpose() * vvector;
                    accumulator.m_gW2.matrix() +=
                        gmatrix.transpose() * (imatrix.array().colwise() * vvector.array()).matrix();
                }
            }
        });

    const auto& accumulator0 = ::nano::sum_reduce(m_accumulators, m_iterator.samples().size());

    // OK, normalize and add the regularization terms
    if (gx != nullptr)
    {
        auto gb = bias(*gx);
        auto gW = weights(*gx);

        gb = accumulator0.m_gb1;
        gW = accumulator0.m_gW1;

        if (m_l1reg > 0.0)
        {
            gW.array() += m_l1reg * W.array().sign() / W.size();
        }
        if (m_l2reg > 0.0)
        {
            gW.array() += m_l2reg * W.array() * 2 / W.size();
        }
        if (m_vAreg > 0.0)
        {
            gb.array() += m_vAreg * (accumulator0.m_gb2.array() - accumulator0.m_vm1 * accumulator0.m_gb1.array()) * 2;
            gW.array() += m_vAreg * (accumulator0.m_gW2.array() - accumulator0.m_vm1 * accumulator0.m_gW1.array()) * 2;
        }
    }

    return accumulator0.m_vm1 + ((m_l1reg > 0.0) ? (m_l1reg * W.array().abs().mean()) : 0.0) +
           ((m_l2reg > 0.0) ? (m_l2reg * W.array().square().mean()) : 0.0) +
           ((m_vAreg > 0.0) ? (m_vAreg * (accumulator0.m_vm2 - accumulator0.m_vm1 * accumulator0.m_vm1)) : 0.0);
}
