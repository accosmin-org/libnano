#include <nano/core/reduce.h>
#include <nano/dataset.h>
#include <nano/linear/function.h>
#include <nano/linear/util.h>

using namespace nano;

namespace
{
auto isize(const flatten_iterator_t& iterator)
{
    return iterator.dataset().columns();
}

auto tsize(const flatten_iterator_t& iterator)
{
    return ::nano::size(iterator.dataset().target_dims());
}
} // namespace

linear::function_t::function_t(const flatten_iterator_t& iterator, const loss_t& loss, scalar_t l1reg, scalar_t l2reg)
    : ::nano::function_t("linear", (::isize(iterator) + 1) * ::tsize(iterator))
    , m_iterator(iterator)
    , m_loss(loss)
    , m_l1reg(l1reg)
    , m_l2reg(l2reg)
    , m_isize(::isize(m_iterator))
    , m_tsize(::tsize(m_iterator))
    , m_accumulators(m_iterator.concurrency(), accumulator_t{m_isize, m_tsize})
{
    assert(m_isize > 0);
    assert(m_tsize > 0);

    convex(m_loss.convex() ? convexity::yes : convexity::no);
    smooth((m_loss.smooth() && m_l1reg <= 0.0) ? smoothness::yes : smoothness::no);
    strong_convexity(m_l2reg / static_cast<scalar_t>(m_isize * m_tsize));
}

rfunction_t linear::function_t::clone() const
{
    return std::make_unique<linear::function_t>(*this);
}

scalar_t linear::function_t::do_eval(eval_t eval) const
{
    const auto b = bias(eval.m_x);
    const auto w = weights(eval.m_x);

    std::for_each(m_accumulators.begin(), m_accumulators.end(), [&](auto& accumulator) { accumulator.clear(); });

    m_iterator.loop(
        [&](tensor_range_t range, size_t tnum, tensor2d_cmap_t inputs, tensor4d_cmap_t targets)
        {
            assert(tnum < m_accumulators.size());
            auto& accumulator = m_accumulators[tnum];

            ::nano::linear::predict(inputs, w, b, accumulator.m_outputs);
            m_loss.value(targets, accumulator.m_outputs, accumulator.m_loss_fx);

            accumulator.m_fx += accumulator.m_loss_fx.sum();

            if (eval.has_grad())
            {
                m_loss.vgrad(targets, accumulator.m_outputs, accumulator.m_loss_gx);

                const auto gmatrix = accumulator.m_loss_gx.reshape(range.size(), m_tsize);
                accumulator.m_gb += gmatrix.matrix().colwise().sum().transpose();
                accumulator.m_gw += gmatrix.matrix().transpose() * inputs;
            }

            if (eval.has_hess())
            {
                m_loss.vhess(targets, accumulator.m_outputs, accumulator.m_loss_hx);

                const auto& hmatrix = accumulator.m_loss_hx;
                // TODO: write it as a more efficient linear algebra operations
                for (tensor_size_t k = 0; k < range.size(); ++k)
                {
                    for (tensor_size_t t1 = 0; t1 < m_tsize; ++t1)
                    {
                        for (tensor_size_t i1 = 0; i1 < m_isize; ++i1)
                        {
                            for (tensor_size_t t2 = 0; t2 < m_tsize; ++t2)
                            {
                                for (tensor_size_t i2 = 0; i2 < m_isize; ++i2)
                                {
                                    accumulator.m_hx(t1 * m_isize + i1, t2 * m_isize + i2) +=
                                        hmatrix(k, t1, 0, 0, t2, 0, 0) * inputs(k, i1) * inputs(k, i2);
                                }
                            }
                        }
                    }

                    for (tensor_size_t t1 = 0; t1 < m_tsize; ++t1)
                    {
                        for (tensor_size_t i1 = 0; i1 < m_isize; ++i1)
                        {
                            for (tensor_size_t t2 = 0; t2 < m_tsize; ++t2)
                            {
                                accumulator.m_hx(t1 * m_isize + i1, m_tsize * m_isize + t2) +=
                                    hmatrix(k, t1, 0, 0, t2, 0, 0) * inputs(k, i1);
                            }
                        }
                    }

                    for (tensor_size_t t1 = 0; t1 < m_tsize; ++t1)
                    {
                        for (tensor_size_t t2 = 0; t2 < m_tsize; ++t2)
                        {
                            for (tensor_size_t i2 = 0; i2 < m_isize; ++i2)
                            {
                                accumulator.m_hx(m_tsize * m_isize + t1, t2 * m_isize + i2) +=
                                    hmatrix(k, t1, 0, 0, t2, 0, 0) * inputs(k, i2);
                            }
                        }
                    }

                    for (tensor_size_t t1 = 0; t1 < m_tsize; ++t1)
                    {
                        for (tensor_size_t t2 = 0; t2 < m_tsize; ++t2)
                        {
                            accumulator.m_hx(m_tsize * m_isize + t1, m_tsize * m_isize + t2) +=
                                hmatrix(k, t1, 0, 0, t2, 0, 0);
                        }
                    }
                }
            }
        });

    const auto& accumulator = ::nano::sum_reduce(m_accumulators, m_iterator.samples().size());

    // OK, normalize and add the regularization terms
    if (eval.has_grad())
    {
        auto gb = bias(eval.m_gx);
        auto gw = weights(eval.m_gx);

        gb = accumulator.m_gb;
        gw = accumulator.m_gw;

        if (m_l1reg > 0.0)
        {
            gw.array() += m_l1reg * w.array().sign() / static_cast<scalar_t>(w.size());
        }
        if (m_l2reg > 0.0)
        {
            gw.array() += m_l2reg * w.array() / static_cast<scalar_t>(w.size());
        }
    }

    if (eval.has_hess())
    {
        eval.m_hx = accumulator.m_hx;

        if (m_l2reg > 0.0)
        {
            eval.m_hx.block(0, 0, w.size(), w.size()).diagonal().array() += m_l2reg / static_cast<scalar_t>(w.size());
        }
    }

    auto fx = accumulator.m_fx;
    if (m_l1reg > 0.0)
    {
        fx += m_l1reg * w.array().abs().mean();
    }
    if (m_l2reg > 0.0)
    {
        fx += 0.5 * (std::sqrt(m_l2reg) * w.array()).square().mean();
    }
    return fx;
}
