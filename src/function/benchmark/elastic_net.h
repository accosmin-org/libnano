#pragma once

#include <function/benchmark/linear.h>
#include <nano/function.h>

namespace nano
{
///
/// \brief empirical risk minimization of loss functions with elastic net regularization:
///     f(x) = 1/2N * sum(loss(W * input_i + b, target_i), i=1,N) + alpha1 * |W| + alpha2/2 * ||W||^2,
///     where x=[W|b].
///
template <class tloss>
class NANO_PUBLIC function_enet_t final : public function_t, private tloss
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_enet_t(tensor_size_t dims = 10, scalar_t alpha1 = 1.0, scalar_t alpha2 = 1.0,
                             tensor_size_t summands = 100);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;

private:
    // attributes
    scalar_t m_alpha1{1.0}; ///< regularization term: L1-norm of the weights
    scalar_t m_alpha2{1.0}; ///< regularization term: squared L2-norm of the weights
};

///
/// \brief mean-squared-error (MSE) loss.
///
class NANO_PUBLIC loss_mse_t : public synthetic_scalar_t
{
public:
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "mse";

    using synthetic_scalar_t::synthetic_scalar_t;

    scalar_t vgrad(matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto delta = outputs - targets;

        if (gx.size() > 0)
        {
            synthetic_linear_t::vgrad(gx, delta);
        }

        return 0.5 * delta.squaredNorm() / static_cast<scalar_t>(outputs.rows());
    }
};

///
/// \brief mean-absolute-error (MAE) loss.
///
class NANO_PUBLIC loss_mae_t : public synthetic_scalar_t
{
public:
    static constexpr auto convex   = true;
    static constexpr auto smooth   = false;
    static constexpr auto basename = "mae";

    using synthetic_scalar_t::synthetic_scalar_t;

    scalar_t vgrad(matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto delta = outputs - targets;

        if (gx.size() > 0)
        {
            synthetic_linear_t::vgrad(gx, delta.array().sign().matrix());
        }

        return delta.array().abs().sum() / static_cast<scalar_t>(outputs.rows());
    }
};

///
/// \brief cauchy loss.
///
class NANO_PUBLIC loss_cauchy_t : public synthetic_scalar_t
{
public:
    static constexpr auto convex   = false;
    static constexpr auto smooth   = false;
    static constexpr auto basename = "cauchy";

    using synthetic_scalar_t::synthetic_scalar_t;

    scalar_t vgrad(matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto delta = outputs - targets;

        if (gx.size() > 0)
        {
            synthetic_linear_t::vgrad(gx, (2.0 * delta.array() / (1.0 + delta.array().square())).matrix());
        }

        return (delta.array().square() + 1.0).log().sum() / static_cast<scalar_t>(outputs.rows());
    }
};

///
/// \brief hinge loss (linear SVM).
///
class NANO_PUBLIC loss_hinge_t : public synthetic_sclass_t
{
public:
    static constexpr auto convex   = true;
    static constexpr auto smooth   = false;
    static constexpr auto basename = "hinge";

    using synthetic_sclass_t::synthetic_sclass_t;

    scalar_t vgrad(matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto edges = -outputs.array() * targets.array();

        if (gx.size() > 0)
        {
            synthetic_linear_t::vgrad(gx, (-targets.array() * ((1.0 + edges).sign() * 0.5 + 0.5)).matrix());
        }

        return (1.0 + edges).max(0.0).sum() / static_cast<scalar_t>(outputs.rows());
    }
};

///
/// \brief logistic loss (binary classification).
///
class NANO_PUBLIC loss_logistic_t : public synthetic_sclass_t
{
public:
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "logistic";

    using synthetic_sclass_t::synthetic_sclass_t;

    scalar_t vgrad(matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto edges = (-outputs.array() * targets.array()).exp();

        if (gx.size() > 0)
        {
            synthetic_linear_t::vgrad(gx, ((-targets.array() * edges) / (1.0 + edges)).matrix());
        }

        return (1.0 + edges).log().sum() / static_cast<scalar_t>(outputs.rows());
    }
};

using function_enet_mae_t      = function_enet_t<loss_mae_t>;
using function_enet_mse_t      = function_enet_t<loss_mse_t>;
using function_enet_hinge_t    = function_enet_t<loss_hinge_t>;
using function_enet_cauchy_t   = function_enet_t<loss_cauchy_t>;
using function_enet_logistic_t = function_enet_t<loss_logistic_t>;
} // namespace nano
