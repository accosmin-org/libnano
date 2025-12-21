#pragma once

#include <function/ml/loss.h>
#include <nano/function.h>

namespace nano
{
///
/// \brief empirical risk minimization of loss functions with lasso regularization:
///     min   f(x) = 1/(2N) * sum(loss(W * input_i + b, target_i), i=1,N) + alpha1 * |W|,
///     where x    = [W|b].
///
/// alternatively the optimization problem can be written as a constrained optimization program:
///     min   f(x) = 1/(2N) * sum(loss(W * input_i + b, target_i), i=1,N) + alpha1 * 1.dot(z),
///     s.t.  -z <= x <= +z (element-wise),
///     where x    = [W|b].
///
/// NB: the number of samples `N` is given as a multiplicative factor `sratio` of the number of free dimensions.
/// NB: only the features with the index multiple of the `modulo` parameter are correlated with the targets.
///
template <class tloss>
class NANO_PUBLIC lasso_function_t final : public function_t, private tloss
{
public:
    ///
    /// \brief constructor
    ///
    explicit lasso_function_t(tensor_size_t dims = 10, uint64_t seed = 42, scalar_t alpha1 = 1.0,
                              scalar_t sratio = 10.0, tensor_size_t modulo = 1);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    string_t do_name() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_eval(eval_t) const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims) const override;

private:
    // attributes
    linear_model_t m_model; ///<
};

using lasso_function_mae_t      = lasso_function_t<loss_mae_t>;
using lasso_function_mse_t      = lasso_function_t<loss_mse_t>;
using lasso_function_hinge_t    = lasso_function_t<loss_hinge_t>;
using lasso_function_cauchy_t   = lasso_function_t<loss_cauchy_t>;
using lasso_function_logistic_t = lasso_function_t<loss_logistic_t>;
} // namespace nano
