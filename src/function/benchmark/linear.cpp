#include <nano/function/benchmark/linear.h>

using namespace nano;

synthetic_linear_t::synthetic_linear_t(tensor_size_t samples, tensor_size_t outputs, tensor_size_t inputs,
                                       tensor_size_t modulo_correlated_inputs)
    : m_inputs(make_random_tensor<scalar_t>(make_dims(samples, inputs), +0.0, +1.0))
    , m_wopt(make_random_tensor<scalar_t>(make_dims(outputs, inputs), +0.0, +1.0))
    , m_bopt(make_random_tensor<scalar_t>(make_dims(outputs), -0.5, +0.5))
{
    for (tensor_size_t o = 0; o < outputs; ++o)
    {
        m_wopt.matrix().row(o) /= m_wopt.matrix().row(o).sum();
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
}

tensor2d_t synthetic_linear_t::outputs(vector_cmap_t x) const
{
    return outputs(make_w(x));
}

tensor2d_t synthetic_linear_t::outputs(matrix_cmap_t w) const
{
    tensor2d_t outputs(m_inputs.size<0>(), m_bopt.size());
    outputs.matrix() = inputs() * w.matrix().transpose();
    outputs.matrix().rowwise() += m_bopt.vector().transpose();
    return outputs;
} // LCOV_EXCL_LINE

synthetic_sclass_t::synthetic_sclass_t(tensor_size_t samples, tensor_size_t outputs, tensor_size_t inputs,
                                       tensor_size_t modulo_correlated_inputs)
    : synthetic_linear_t(samples, outputs, inputs, modulo_correlated_inputs)
    , m_targets(samples, outputs)
{
    const auto xoutputs = this->outputs(wopt());
    for (tensor_size_t s = 0; s < samples; ++s)
    {
        const auto woutput        = xoutputs.matrix().array().row(s) - bopt().array();
        m_targets.matrix().row(s) = (woutput - 0.5).sign();
    }
}

synthetic_scalar_t::synthetic_scalar_t(tensor_size_t samples, tensor_size_t outputs, tensor_size_t inputs,
                                       tensor_size_t modulo_correlated_inputs)
    : synthetic_linear_t(samples, outputs, inputs, modulo_correlated_inputs)
    , m_targets(this->outputs(wopt()))
{
}
