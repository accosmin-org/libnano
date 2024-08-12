#pragma once

#include <nano/learner.h>
#include <nano/linear/function.h>
#include <nano/mlearn/params.h>
#include <nano/mlearn/result.h>

namespace nano
{
class linear_t;
using rlinear_t = std::unique_ptr<linear_t>;

///
/// \brief a linear model is an affine transformation of the flatten input features x:
///     y(x) = weights * x + bias.
///
/// NB: the training is performed using generic loss functions
///     (e.g. hinge loss, logistic loss, squared error, absolute error)
///     and as such these models generalize the standard linear models
///     that use mean squared error (MSE) like ordinary least squares, lasso, ridge regression or elastic net.
///
/// NB: thus these models can be used for both:
///     - classification (both binary and multi-class) and
///     - regression (both univariate and multivariate) depending on the chosen loss function.
///
/// NB: the inputs should be normalized during training to speed-up convergence (@see nano::scaling_type).
///
/// see "Regression Shrinkage and Selection via the lasso", by R. Tibshirani
/// see "Regularization and variable selection via the elastic net", by H. Zou, T. Hastie
/// see "Statistical Learning with Sparsity: The Lasso and Generalizations", by T. Hastie, R. Tibshirani, M. Mainwright
/// see "The Elements of Statistical Learning", by T. Hastie, R. Tibshirani
///
class NANO_PUBLIC linear_t : public typed_t, public learner_t, public clonable_t<linear_t>
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_t(string_t = "linear");

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<linear_t>& all();

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

    ///
    /// \brief returns the hyper-parameters to tune.
    ///
    virtual param_spaces_t make_param_spaces() const = 0;

    ///
    /// \brief returns the loss function to optimize for the given hyper-parameters values.
    ///
    virtual linear::function_t make_function(const flatten_iterator_t&, const loss_t&, tensor1d_cmap_t) const = 0;

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
