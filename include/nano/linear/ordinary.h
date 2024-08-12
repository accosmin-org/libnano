#pragma once

#include <nano/linear.h>

namespace nano
{
///
/// \brief a generalization of ordinary least squares (OLS) to generic loss functions:
///     given a linear model:
///         y(x) = weights * x + bias,
///
///     the training is performed using the following criterion:
///         L(weights, bias) = 1/N sum(loss(y(x_i), y_i), i=1..N).
///
/// NB: the training criterion is convex if the loss function is convex as well.
///
/// NB: the traditional OLS algorithm is retrieved if the loss function is the squared error.
///
class NANO_PUBLIC ordinary_t final : public linear_t
{
public:
    ///
    /// \brief constructor
    ///
    ordinary_t();

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
