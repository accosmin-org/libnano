#pragma once

#include <nano/model.h>

namespace nano
{
    ///
    /// \brief a linear model is an affine transformation of the flatten input features x:
    ///     y(x) = weights * x + bias.
    ///
    /// NB: the model can be regularized using the following methods:
    ///     - lasso (the L1-norm of the weights),
    ///     - ridge (the L2-norm of the weights),
    ///     - elastic net (the L1-norm and the L2-norm of the weights) or
    ///     - variance (of the loss values across samples).
    ///
    /// NB: the inputs should be normalized during training to speed-up convergence (@see nano::scaling_type).
    ///
    /// see "Regression Shrinkage and Selection via the lasso", by R. Tibshirani
    /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
    /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
    ///
    class NANO_PUBLIC linear_model_t final : public model_t
    {
    public:
        ///
        /// \brief constructor
        ///
        linear_model_t();

        ///
        /// \brief @see configurable_t
        ///
        std::istream& read(std::istream&) override;

        ///
        /// \brief @see configurable_t
        ///
        std::ostream& write(std::ostream&) const override;

        ///
        /// \brief @see clone_t
        ///
        rmodel_t clone() const override;

        ///
        /// \brief @see model_
        ///
        fit_result_t fit(const dataset_t&, const indices_t&, const loss_t&, const solver_t&, const splitter_t&,
                         const tuner_t&) override;

        ///
        /// \brief @see model_t
        ///
        tensor4d_t predict(const dataset_t&, const indices_t&) const override;

        ///
        /// \brief returns the fitted bias vector (intercept).
        ///
        const tensor1d_t& bias() const { return m_bias; }

        ///
        /// \brief returns the fitted weigths matrix (coefficients).
        ///
        const tensor2d_t& weights() const { return m_weights; }

    private:
        // attributes
        tensor1d_t m_bias;    ///< bias vector (#outputs)
        tensor2d_t m_weights; ///< weight matrix (#inputs, #outputs)
    };
} // namespace nano
