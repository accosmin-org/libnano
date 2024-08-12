#pragma once

#include <nano/linear.h>

namespace nano
{
///
/// \brief a generalization of ELASTIC NET to generic loss functions:
///     given a linear model:
///         y(x) = weights * x + bias,
///
///     the training is performed using the following criterion:
///         L(weights, bias) = 1/N sum(loss(y(x_i), y_i), i=1..N) + alpha * |weights| + beta * 1/2 * ||weights||^2.
///
/// NB: the training criterion is convex if the loss function is convex as well.
///
/// NB: the training criterion is strongly convex if the loss function is convex and beta is strictly positive.
///
/// NB: the traditional ELASTIC NET algorithm is retrieved if the loss function is the squared error.
///
/// NB: the regularization terms that penalizes the L1 and L2-norm of the weights are data-dependent
///     and need to be tuned during training.
///
/// NB: this model combines the benefits of both LASSO and RIDGE regression models:
///     sparse solutions (useful for feature selection) and with small coefficients (to generalize better).
///
class NANO_PUBLIC elastic_net_t final : public linear_t
{
public:
    ///
    /// \brief constructor
    ///
    elastic_net_t();

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
