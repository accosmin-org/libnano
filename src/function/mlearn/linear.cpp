#include <function/mlearn/linear.h>

using namespace nano;

linear_model_t::linear_model_t(const tensor_size_t samples, const tensor_size_t outputs, const tensor_size_t inputs,
                               const uint64_t seed, const tensor_size_t modulo_correlated_inputs, const bool regression)
    : m_inputs(samples, inputs)
    , m_targets(samples, outputs)
    , m_outputs(samples, outputs)
    , m_gradbuffs(samples, outputs)
    , m_hessbuffs(samples, outputs, outputs)
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

    eval_outputs(m_woptimum);

    if (regression)
    {
        m_targets = m_outputs;
    }
    else
    {
        m_targets = ((m_outputs.matrix().array().rowwise() - m_boptimum.transpose().array()) - 0.5).sign();
    }
}

void linear_model_t::eval_outputs(const vector_cmap_t x) const
{
    eval_outputs(make_w(x));
}

void linear_model_t::eval_outputs(const matrix_cmap_t w) const
{
    m_outputs.matrix().noalias() = m_inputs * w.transpose();
    m_outputs.matrix().rowwise() += m_boptimum.transpose();
}

void linear_model_t::eval_grad(vector_map_t gx) const
{
    const auto samples = m_gradbuffs.rows();

    auto gw = make_w(gx).matrix();

    // cppcheck-suppress redundantInitialization
    // cppcheck-suppress unreadVariable
    gw.noalias() = m_gradbuffs.matrix().transpose() * m_inputs.matrix() / static_cast<scalar_t>(samples);
}

void linear_model_t::eval_hess(matrix_map_t Hx) const
{
    const auto inputs  = m_inputs.matrix();
    const auto samples = m_gradbuffs.rows();
    const auto outputs = m_gradbuffs.cols();

    assert(outputs == 1);

    Hx.matrix().noalias() = (inputs.array().colwise() * m_hessbuffs.array()).matrix().transpose() * inputs;

    // NB: multivariate version, but harder to guarantee the expected number of free dimensions!
    // TODO: write the following operations using Eigen3 calls for improved performance
    /*
    const auto nparams = m_woptimum.size();
    Hx.full(0.0);
    for (tensor_size_t i = 0; i < nparams; ++i)
    {
        for (tensor_size_t j = 0; j < nparams; ++j)
        {
            for (tensor_size_t sample = 0; sample < samples; ++sample)
            {
                Hx(i, j) += m_hessbuffs(sample, i % outputs, j % outputs) * m_inputs(sample, i / outputs) *
                            m_inputs(sample, j / outputs);
            }
        }
    }*/

    Hx.array() /= static_cast<scalar_t>(samples);
}
