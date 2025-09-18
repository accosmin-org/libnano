#pragma once

#include <nano/tensor.h>

namespace nano
{
///
/// \brief synthetic linear machine learning model
///     where the predictions are an affine transformation of the inputs.
///
/// NB: the targets can be configured to be correlated only to some inputs (features) modulo a fixed constant.
/// NB: simulates either univariate regression or classification problems.
///
class NANO_PUBLIC linear_model_t
{
public:
    linear_model_t(tensor_size_t samples, tensor_size_t outputs, tensor_size_t inputs, uint64_t seed,
                   tensor_size_t modulo_correlated_inputs, bool regression);

    const matrix_t& wopt() const { return m_woptimum; }

    const vector_t& bopt() const { return m_boptimum; }

    const matrix_t& inputs() const { return m_inputs; }

    const matrix_t& targets() const { return m_targets; }

    const matrix_t& outputs() const { return m_outputs; }

    const matrix_t& outputs(vector_cmap_t x) const;

    const matrix_t& outputs(matrix_cmap_t w) const;

    matrix_map_t gradients() const { return m_gradients.tensor(); }

    tensor3d_map_t hessians() const { return m_hessians.tensor(); }

    matrix_map_t make_w(vector_map_t x) const { return map_tensor(x.data(), m_woptimum.dims()); }

    matrix_cmap_t make_w(vector_cmap_t x) const { return map_tensor(x.data(), m_woptimum.dims()); }

    bool eval_grad(vector_map_t gx) const;

    bool eval_hess(matrix_map_t Hx) const;

private:
    matrix_t           m_inputs;    ///< inputs (#samples, #inputs)
    matrix_t           m_targets;   ///< targets (#samples, #outputs: univariate regression or classification)
    mutable matrix_t   m_outputs;   ///< outputs (#samples, #outputs)
    mutable matrix_t   m_gradients; ///< gradients of the loss wrt output (#samples, #outputs)
    mutable tensor3d_t m_hessians;  ///< hessians of the loss wrt output (#samples, #outputs, #outputs)
    matrix_t           m_woptimum;  ///< weights used for generating the synthetic dataset
    vector_t           m_boptimum;  ///< bias used for generating the synthetic dataset
};
} // namespace nano
