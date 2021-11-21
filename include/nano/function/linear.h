#pragma once

#include <nano/tensor.h>

namespace nano
{
    ///
    /// \brief synthetic linear machine model where the predictions are am affine transformation of the inputs.
    ///
    /// NB: the targets can be configured to be correlated only to some inputs (features) modulo a fixed constant.
    ///
    class synthetic_linear_t
    {
    public:

        synthetic_linear_t(
            tensor_size_t samples, tensor_size_t outputs, tensor_size_t inputs,
            tensor_size_t modulo_correlated_inputs = 1);

        const auto& wopt() const { return m_wopt; }
        const auto& bopt() const { return m_bopt; }
        auto inputs() const { return m_inputs.matrix(); }
        auto inputs(tensor_size_t summand) const { return m_inputs.slice(summand, summand + 1).matrix(); }

        tensor2d_t outputs(const vector_t& x) const;
        tensor2d_t outputs(tensor2d_cmap_t w, tensor1d_cmap_t b) const;

        tensor2d_t outputs(const vector_t& x, tensor_size_t summand) const;
        tensor2d_t outputs(tensor2d_cmap_t w, tensor1d_cmap_t b, tensor_size_t summand) const;

        auto make_w(vector_t& x) const { return map_tensor(x.data(), m_wopt.dims()); }
        auto make_w(const vector_t& x) const { return map_tensor(x.data(), m_wopt.dims()); }

        auto make_b(vector_t& x) const { return map_tensor(x.data() + m_wopt.size(), m_bopt.dims()); }
        auto make_b(const vector_t& x) const { return map_tensor(x.data() + m_wopt.size(), m_bopt.dims()); }

        template <typename tgrad, typename tinputs>
        void vgrad(vector_t* gx, const tgrad& gg, const tinputs& inputs) const
        {
            const auto samples = static_cast<scalar_t>(gg.rows());

            auto gw = make_w(*gx).matrix();
            auto gb = make_b(*gx).vector();

            // cppcheck-suppress redundantInitialization
            // cppcheck-suppress unreadVariable
            gw = gg.transpose() * inputs.matrix() / samples;

            // cppcheck-suppress redundantInitialization
            // cppcheck-suppress unreadVariable
            gb = gg.colwise().sum() / samples;
        }

    private:

        // attributes
        tensor2d_t  m_inputs;           ///<
        tensor2d_t  m_wopt;             ///< weights used for generating the synthetic dataset
        tensor1d_t  m_bopt;             ///< bias used for generating the synthetic dataset
    };

    ///
    /// \brief synthetic binary classification with a linear model.
    ///
    class synthetic_sclass_t : public synthetic_linear_t
    {
    public:

        synthetic_sclass_t(
            tensor_size_t samples, tensor_size_t outputs, tensor_size_t inputs,
            tensor_size_t modulo_correlated_inputs = 1);

        auto targets() const { return m_targets.matrix(); }
        auto targets(tensor_size_t summand) const { return m_targets.slice(summand, summand + 1).matrix(); }

    private:

        tensor2d_t  m_targets;          ///<
    };

    ///
    /// \brief synthetic univariate regression with a linear model.
    ///
    class synthetic_scalar_t : public synthetic_linear_t
    {
    public:

        synthetic_scalar_t(
            tensor_size_t samples, tensor_size_t outputs, tensor_size_t inputs,
            tensor_size_t modulo_correlated_inputs = 1);

        auto targets() const { return m_targets.matrix(); }
        auto targets(tensor_size_t summand) const { return m_targets.slice(summand, summand + 1).matrix(); }

    private:

        tensor2d_t  m_targets;          ///<
    };
}
