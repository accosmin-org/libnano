#pragma once

#include <nano/linear.h>

namespace nano
{
///
/// \brief a generalization of RIDGE regression to generic loss functions:
///     given a linear model:
///         y(x) = weights * x + bias,
///
///     the training is performed using the following criterion:
///         L(weights, bias) = 1/N sum(loss(y(x_i), y_i), i=1..N) + beta * 1/2 * ||weights||^2.
///
/// NB: the training criterion is convex if the loss function is convex as well.
///
/// NB: the traditional RIDGE regression algorithm is retrieved if the loss function is the squared error.
///
/// NB: the regularization term that penalizes the L2-norm of the weights is data-dependent
///     and needs to be tuned during training.
///
/// NB: the higher the regularization parameter, the smaller the magnitude of the coefficients.
///
class NANO_PUBLIC ridge_t final : public linear_t
{
public:
    ///
    /// \brief constructor
    ///
    ridge_t();

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
