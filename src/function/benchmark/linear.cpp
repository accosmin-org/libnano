#include <function/benchmark/linear.h>

using namespace nano;

linear_model_t::linear_model_t(const tensor_size_t samples, const tensor_size_t outputs, const tensor_size_t inputs,
                               const tensor_size_t modulo_correlated_inputs, const bool regression)
    : m_inputs(make_random_tensor<scalar_t>(make_dims(samples, inputs), +0.0, +1.0, seed_t{42}))
    , m_targets(samples, outputs)
    , m_wopt(make_random_tensor<scalar_t>(make_dims(outputs, inputs), +0.0, +1.0, seed_t{42}))
    , m_bopt(make_random_tensor<scalar_t>(make_dims(outputs), -0.5, +0.5, seed_t{42}))
{
    for (tensor_size_t o = 0; o < outputs; ++o)
    {
        m_wopt.row(o) /= m_wopt.row(o).sum();
    }
    for (tensor_size_t i = 0; i < inputs; ++i)
    {
        m_inputs.matrix().col(i) /= m_inputs.matrix().col(i).sum();
    }

    for (tensor_size_t i = 0; i < inputs; ++i)
    {
        if (i % modulo_correlated_inputs != 0)
        {
            m_wopt.matrix().array().col(i) = 0.0;
        }
    }

    if (const auto xoutputs = this->outputs(wopt()); regression)
    {
        m_targets = xoutputs;
    }
    else
    {
        for (tensor_size_t s = 0; s < samples; ++s)
        {
            const auto woutput = xoutputs.matrix().array().row(s) - bopt().array();
            m_targets.row(s)   = (woutput - 0.5).sign();
        }
    }
}

tensor2d_t linear_model_t::outputs(vector_cmap_t x) const
{
    return outputs(make_w(x));
}

tensor2d_t linear_model_t::outputs(matrix_cmap_t w) const
{
    tensor2d_t outputs(m_inputs.size<0>(), m_bopt.size());
    outputs.matrix() = inputs().matrix() * w.transpose();
    outputs.matrix().rowwise() += m_bopt.transpose();
    return outputs;
}
