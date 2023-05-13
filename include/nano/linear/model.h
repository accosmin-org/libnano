#pragma once

#include <nano/learner.h>
#include <nano/loss.h>
#include <nano/mlearn/params.h>
#include <nano/mlearn/result.h>

namespace nano
{
///
/// \brief a linear model is an affine transformation of the flatten input features x:
///     y(x) = weights * x + bias.
///
/// NB: the model can be regularized using the following methods:
///     - lasso (the L1-norm of the weights),
///     - ridge (the L2-norm of the weights),
///     - elastic net (the L1-norm and the L2-norm of the weights).
///
/// NB: the inputs should be normalized during training to speed-up convergence (@see nano::scaling_type).
///
/// see "Regression Shrinkage and Selection via the lasso", by R. Tibshirani
/// see "Regularization and variable selection via the elastic net", by H. Zou, T. Hastie
///
class NANO_PUBLIC linear_model_t final : public learner_t
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
    /// \brief fit the model using the given samples and return the associated statistics.
    ///
    ml::result_t fit(const dataset_t&, const indices_t&, const loss_t&, const ml::params_t& = {});

    ///
    /// \brief returns the fitted bias vector (intercept).
    ///
    const tensor1d_t& bias() const { return m_bias; }

    ///
    /// \brief returns the fitted weigths matrix (coefficients).
    ///
    const tensor2d_t& weights() const { return m_weights; }

private:
    ///
    /// \brief @see learner_t
    ///
    void do_predict(const dataset_t&, indices_cmap_t, tensor4d_map_t) const override;

    // attributes
    tensor1d_t m_bias;    ///< bias vector (#outputs)
    tensor2d_t m_weights; ///< weight matrix (#inputs, #outputs)
};
} // namespace nano
