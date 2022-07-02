#include <nano/model/surrogate.h>

using namespace nano;

quadratic_surrogate_fit_t::quadratic_surrogate_fit_t(const loss_t& loss, tensor2d_t p, tensor1d_t y)
    : function_t("quadratic surrogate fitting function", (p.cols() + 1) * (p.cols() + 2) / 2)
    , m_loss(loss)
    , m_p2(p.size<0>(), size())
    , m_y(std::move(y))
    , m_loss_outputs(p.size<0>(), 1, 1, 1)
    , m_loss_values(p.size<0>())
    , m_loss_vgrads(p.size<0>(), 1, 1, 1)
{
    convex(loss.convex());
    smooth(loss.smooth());

    assert(m_p2.size<0>() == m_y.size<0>());

    for (tensor_size_t sample = 0, samples = p.size<0>(), size = p.size<1>(); sample < samples; ++sample)
    {
        auto k = tensor_size_t{0};

        m_p2(sample, k++) = 1.0;
        for (tensor_size_t i = 0; i < size; ++i)
        {
            m_p2(sample, k++) = p(sample, i);
        }
        for (tensor_size_t i = 0; i < size; ++i)
        {
            for (tensor_size_t j = i; j < size; ++j)
            {
                m_p2(sample, k++) = p(sample, i) * p(sample, j);
            }
        }
    }
}

scalar_t quadratic_surrogate_fit_t::vgrad(const vector_t& x, vector_t* gx) const
{
    m_loss_outputs.vector() = m_p2.matrix() * x;

    const auto samples = m_p2.size<0>();
    const auto targets = m_y.reshape(samples, 1, 1, 1);

    if (gx != nullptr)
    {
        m_loss.vgrad(targets, m_loss_outputs, m_loss_vgrads);
        gx->noalias() = m_p2.matrix().transpose() * m_loss_vgrads.vector();
    }

    m_loss.value(targets, m_loss_outputs, m_loss_values);
    return m_loss_values.sum();
}

quadratic_surrogate_t::quadratic_surrogate_t(vector_t model)
    : function_t("quadratic surrogate function", static_cast<tensor_size_t>(std::sqrt(2 * model.size())) - 1)
    , m_model(std::move(model))
{
    convex(false);
    smooth(true);

    assert(size() > 0);
    assert(m_model.size() == (size() + 1) * (size() + 2) / 2);
}

scalar_t quadratic_surrogate_t::vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        auto& g = *gx;
        g.setZero();

        auto k = tensor_size_t{1};
        for (tensor_size_t i = 0, size = x.size(); i < size; ++i)
        {
            g(i) += m_model(k++);
        }
        for (tensor_size_t i = 0, size = x.size(); i < size; ++i)
        {
            for (tensor_size_t j = i; j < size; ++j)
            {
                g(i) += m_model(k) * x(j);
                g(j) += m_model(k++) * x(i);
            }
        }
    }

    scalar_t fx = m_model(0);

    auto k = tensor_size_t{1};
    for (tensor_size_t i = 0, size = x.size(); i < size; ++i)
    {
        fx += m_model(k++) * x(i);
    }
    for (tensor_size_t i = 0, size = x.size(); i < size; ++i)
    {
        for (tensor_size_t j = i; j < size; ++j)
        {
            fx += m_model(k++) * x(i) * x(j);
        }
    }
    return fx;
}
