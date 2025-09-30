#pragma once

#include <nano/function.h>

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

    template <class tloss>
    scalar_t eval(function_t::eval_t eval) const
    {
        eval_outputs(eval.m_x);

        if (eval.has_grad())
        {
            tloss::gx(m_outputs, m_targets, m_gradbuffs);
            eval_grad(eval.m_gx);
        }

        if (eval.has_hess())
        {
            tloss::hx(m_outputs, m_targets, m_hessbuffs);
            eval_hess(eval.m_hx);
        }

        return tloss::fx(m_outputs, m_targets);
    }

private:
    void eval_grad(vector_map_t gx) const;
    void eval_hess(matrix_map_t hx) const;
    void eval_outputs(vector_cmap_t x) const;
    void eval_outputs(matrix_cmap_t w) const;

    matrix_map_t make_w(vector_map_t x) const { return map_tensor(x.data(), m_woptimum.dims()); }

    matrix_cmap_t make_w(vector_cmap_t x) const { return map_tensor(x.data(), m_woptimum.dims()); }

    matrix_t           m_inputs;    ///< inputs (#samples, #inputs)
    matrix_t           m_targets;   ///< targets (#samples, #outputs: univariate regression or classification)
    mutable matrix_t   m_outputs;   ///< outputs (#samples, #outputs)
    mutable matrix_t   m_gradbuffs; ///< gradients of the loss wrt output (#samples, #outputs)
    mutable tensor3d_t m_hessbuffs; ///< hessians of the loss wrt output (#samples, #outputs, #outputs)
    matrix_t           m_woptimum;  ///< weights used for generating the synthetic dataset
    vector_t           m_boptimum;  ///< bias used for generating the synthetic dataset
};
} // namespace nano
