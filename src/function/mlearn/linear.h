#pragma once

#include <nano/tensor.h>

namespace nano
{
///
/// \brief synthetic linear machine learning model
///     where the predictions are am affine transformation of the inputs.
///
/// NB: the targets can be configured to be correlated only to some inputs (features) modulo a fixed constant.
/// NB: simulates either univariate regression or classification problems.
///
class NANO_PUBLIC linear_model_t
{
public:
    linear_model_t(tensor_size_t samples, tensor_size_t outputs, tensor_size_t inputs, uint64_t seed,
                   tensor_size_t modulo_correlated_inputs, bool regression);

    const matrix_t& wopt() const { return m_wopt; }

    const vector_t& bopt() const { return m_bopt; }

    const matrix_t& inputs() const { return m_inputs; }

    const matrix_t& targets() const { return m_targets; }

    tensor2d_t outputs(vector_cmap_t x) const;

    tensor2d_t outputs(matrix_cmap_t w) const;

    matrix_map_t make_w(vector_map_t x) const { return map_tensor(x.data(), m_wopt.dims()); }

    matrix_cmap_t make_w(vector_cmap_t x) const { return map_tensor(x.data(), m_wopt.dims()); }

    template <class tgrad, class thess>
    void eval(vector_map_t gx, matrix_map_t Hx, const tgrad& gg, const thess& hh) const
    {
        const auto size    = gg.cols();
        const auto samples = static_cast<scalar_t>(gg.rows());

        // TODO:
        gx(i)    = sum_k(gg(k, i) * inputs(k, i));
        Hx(i, j) = sum_k(hh(k, i, j) * inputs(k, i) * inputs(k, j));

        if (gx.size() == size())
        {
            auto gw = make_w(gx).matrix();

            // cppcheck-suppress redundantInitialization
            // cppcheck-suppress unreadVariable
            gw = gg.matrix().transpose() * m_inputs.matrix() / samples;
        }

        if (Hx.rows() == size() && Hx.cols() == size())
        {
        }
    }

private:
    matrix_t m_inputs;  ///< inputs
    matrix_t m_targets; ///< targets (univariate regression or classification)
    matrix_t m_wopt;    ///< weights used for generating the synthetic dataset
    vector_t m_bopt;    ///< bias used for generating the synthetic dataset
};
} // namespace nano
