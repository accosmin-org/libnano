#include <function/mlearn/linear.h>

using namespace nano;

linear_model_t::linear_model_t(const tensor_size_t samples, const tensor_size_t outputs, const tensor_size_t inputs,
                               const uint64_t seed, const tensor_size_t modulo_correlated_inputs, const bool regression)
    : m_inputs(samples, inputs)
    , m_targets(samples, outputs)
    , m_outputs(samples, outputs)
    , m_gradients(samples, outputs)
    , m_hessians(samples, outputs, outputs)
    , m_woptimum(outputs, inputs)
    , m_boptimum(outputs)
{
    auto rng   = make_rng(seed);
    auto udist = make_udist<scalar_t>(0.0, 1.0);

    m_inputs.full([&]() { return udist(rng); });
    m_woptimum.full([&]() { return udist(rng); });
    m_boptimum.full([&]() { return udist(rng) - 0.5; });

    for (tensor_size_t o = 0; o < outputs; ++o)
    {
        m_woptimum.row(o) /= m_woptimum.row(o).sum();
    }

    for (tensor_size_t i = 0; i < inputs; ++i)
    {
        if (i % modulo_correlated_inputs != 0)
        {
            m_woptimum.matrix().array().col(i) = 0.0;
        }
    }

    m_outputs.matrix() = m_inputs * m_woptimum.transpose();
    m_outputs.matrix().rowwise() += m_boptimum.transpose();

    if (regression)
    {
        m_targets = m_outputs;
    }
    else
    {
        m_targets = ((m_outputs.matrix().array().rowwise() - m_boptimum.transpose().array()) - 0.5).sign();
    }
}

const tensor2d_t& linear_model_t::outputs(const vector_cmap_t x) const
{
    return outputs(make_w(x));
}

const tensor2d_t& linear_model_t::outputs(const matrix_cmap_t w) const
{
    m_outputs.matrix() = m_inputs * w.transpose();
    m_outputs.matrix().rowwise() += m_boptimum.transpose();
    return m_outputs;
}

bool linear_model_t::eval_grad(vector_map_t gx) const
{
    const auto [samples, outputs] = m_gradients.dims();

    if (gx.size() == m_woptimum.size())
    {
        auto gw = make_w(gx).matrix();

        // cppcheck-suppress redundantInitialization
        // cppcheck-suppress unreadVariable
        gw = m_gradients.matrix().transpose() * m_inputs.matrix();
        gw.array() /= static_cast<scalar_t>(samples);
        return true;
    }
    else
    {
        return false;
    }
}

bool linear_model_t::eval_hess(matrix_map_t Hx) const
{
    const auto [samples, outputs] = m_gradients.dims();

    if (Hx.rows() == outputs && Hx.cols() == outputs)
    {
        Hx.full(0.0);

        // TODO: write the following operations using Eigen3 calls for improved performance
        // Hx(i, j) = sum_k(hh(k, i, j) * inputs(k, i) * inputs(k, j));
        return true;
    }
    else
    {
        return false;
    }
}
