#pragma once

#include <nano/linear.h>

namespace nano
{
///
/// \brief a generalization of LASSO to generic loss functions:
///     given a linear model:
///         y(x) = weights * x + bias,
///
///     the training is performed using the following criterion:
///         L(weights, bias) = 1/N sum(loss(y(x_i), y_i), i=1..N) + alpha * |weights|.
///
/// NB: the training criterion is convex if the loss function is convex as well.
///
/// NB: the traditional LASSO algorithm is retrieved if the loss function is the squared error.
///
/// NB: the regularization term that penalizes the L1-norm of the weights is data-dependent
///     and needs to be tuned during training.
///
/// NB: the higher the regularization parameter, the sparser the solution (e.g. the more coefficients close to zero)
///     and thus the model can be used for feature selection as well.
///
class NANO_PUBLIC lasso_t final : public linear_t
{
public:
    ///
    /// \brief constructor
    ///
    lasso_t();

    ///
    /// @see clonable_t
    ///
    rlinear_t clone() const override;

    ///
    /// @see linear_t
    ///
    param_spaces_t make_param_spaces() const override;

    ///
    /// @see linear_t
    ///
    linear::function_t make_function(const flatten_iterator_t&, const loss_t&, tensor1d_cmap_t) const override;
};
} // namespace nano
