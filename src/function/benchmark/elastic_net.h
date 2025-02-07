#pragma once

#include <function/benchmark/linear.h>
#include <nano/function.h>

namespace nano
{
///
/// \brief empirical risk minimization of loss functions with elastic net regularization:
///     f(x)    = 1/(2N) * sum(loss(W * input_i + b, target_i), i=1,N) + alpha1 * |W| + alpha2/2 * ||W||^2,
///     where x = [W|b].
///
template <class tloss>
class NANO_PUBLIC function_enet_t final : public function_t, private tloss
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_enet_t(tensor_size_t dims = 10, scalar_t alpha1 = 0.0, scalar_t alpha2 = 0.0,
                             scalar_t sample_ratio = 10.0, tensor_size_t modulo_correlated = 1);

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
    rfunction_t make(tensor_size_t dims) override;

private:
    // attributes
    linear_model_t m_model; ///<
};

///
/// \brief mean-squared-error (MSE) loss.
///
class NANO_PUBLIC loss_mse_t
{
public:
    static constexpr auto convex     = true;
    static constexpr auto smooth     = true;
    static constexpr auto basename   = "mse";
    static constexpr auto regression = true;

    scalar_t vgrad(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto delta = outputs - targets;

        if (gx.size() > 0)
        {
            model.vgrad(gx, delta);
        }

        return 0.5 * delta.squaredNorm() / static_cast<scalar_t>(outputs.rows());
    }
};

///
/// \brief mean-absolute-error (MAE) loss.
///
class NANO_PUBLIC loss_mae_t
{
public:
    static constexpr auto convex     = true;
    static constexpr auto smooth     = false;
    static constexpr auto basename   = "mae";
    static constexpr auto regression = true;

    scalar_t vgrad(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto delta = outputs - targets;

        if (gx.size() > 0)
        {
            model.vgrad(gx, delta.array().sign().matrix());
        }

        return delta.array().abs().sum() / static_cast<scalar_t>(outputs.rows());
    }
};

///
/// \brief cauchy loss.
///
class NANO_PUBLIC loss_cauchy_t
{
public:
    static constexpr auto convex     = false;
    static constexpr auto smooth     = false;
    static constexpr auto basename   = "cauchy";
    static constexpr auto regression = true;

    scalar_t vgrad(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto delta = outputs - targets;

        if (gx.size() > 0)
        {
            model.vgrad(gx, (2.0 * delta.array() / (1.0 + delta.array().square())).matrix());
        }

        return (delta.array().square() + 1.0).log().sum() / static_cast<scalar_t>(outputs.rows());
    }
};

///
/// \brief hinge loss (linear SVM).
///
class NANO_PUBLIC loss_hinge_t
{
public:
    static constexpr auto convex     = true;
    static constexpr auto smooth     = false;
    static constexpr auto basename   = "hinge";
    static constexpr auto regression = false;

    scalar_t vgrad(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto edges = -outputs.array() * targets.array();

        if (gx.size() > 0)
        {
            model.vgrad(gx, (-targets.array() * ((1.0 + edges).sign() * 0.5 + 0.5)).matrix());
        }

        return (1.0 + edges).max(0.0).sum() / static_cast<scalar_t>(outputs.rows());
    }
};

///
/// \brief logistic loss (binary classification).
///
class NANO_PUBLIC loss_logistic_t
{
public:
    static constexpr auto convex     = true;
    static constexpr auto smooth     = true;
    static constexpr auto basename   = "logistic";
    static constexpr auto regression = false;

    scalar_t vgrad(const linear_model_t& model, matrix_cmap_t outputs, matrix_cmap_t targets, vector_map_t gx) const
    {
        const auto edges = (-outputs.array() * targets.array()).exp();

        if (gx.size() > 0)
        {
            model.vgrad(gx, ((-targets.array() * edges) / (1.0 + edges)).matrix());
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
