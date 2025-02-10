#pragma once

#include <function/mlearn/loss.h>
#include <nano/function.h>

namespace nano
{
///
/// \brief empirical risk minimization of loss functions with ridge regularization:
///     f(x)    = 1/(2N) * sum(loss(W * input_i + b, target_i), i=1,N) + alpha2/2 * ||W||^2,
///     where x = [W|b].
///
/// NB: the number of samples `N` is given as a multiplicative factor `sratio` of the number of free dimensions.
/// NB: only the features with the index multiple of the `modulo` parameter are correlated with the targets.
///
template <class tloss>
class NANO_PUBLIC function_ridge_t final : public function_t, private tloss
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_ridge_t(tensor_size_t dims = 10, scalar_t alpha2 = 0.0, scalar_t sratio = 10.0,
                              tensor_size_t modulo = 1);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    string_t name(bool with_size = true) const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims) const override;

private:
    // attributes
    linear_model_t m_model; ///<
};

using function_ridge_mae_t      = function_ridge_t<loss_mae_t>;
using function_ridge_mse_t      = function_ridge_t<loss_mse_t>;
using function_ridge_hinge_t    = function_ridge_t<loss_hinge_t>;
using function_ridge_cauchy_t   = function_ridge_t<loss_cauchy_t>;
using function_ridge_logistic_t = function_ridge_t<loss_logistic_t>;
} // namespace nano
