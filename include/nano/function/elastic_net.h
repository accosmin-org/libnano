#pragma once

#include <nano/function/linear.h>
#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief empirical risk minimization of loss functions with elastic net regularization:
    ///     f(x) = 1/2N * sum(loss(W * input_i + b, target_i), i=1,N) + alpha1 * |W| + alpha2/2 * ||W||^2,
    ///     where x=[W|b].
    ///
    template <typename tloss>
    class NANO_PUBLIC function_enet_t final : public benchmark_function_t, private tloss
    {
    public:

        ///
        /// \brief constructor
        ///
        explicit function_enet_t(tensor_size_t dims = 10, scalar_t alpha1 = 1.0, scalar_t alpha2 = 1.0);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx) const override;

        ///
        /// \brief @see benchmark_function_t
        ///
        rfunction_t make(tensor_size_t dims) const override;

    private:

        // attributes
        scalar_t    m_alpha1{1.0};  ///< regularization term: L1-norm of the weights
        scalar_t    m_alpha2{1.0};  ///< regularization term: squared L2-norm of the weights
    };

    ///
    /// \brief mean-squared-error (MSE) loss.
    ///
    class NANO_PUBLIC loss_mse_t : public synthetic_scalar_t
    {
    public:

        static constexpr auto smooth = true;
        static constexpr auto basename = "MSE";

        ///
        /// \brief constructor
        ///
        explicit loss_mse_t(tensor_size_t dims);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx) const;
    };

    ///
    /// \brief mean-absolute-error (MAE) loss.
    ///
    class NANO_PUBLIC loss_mae_t : public synthetic_scalar_t
    {
    public:

        static constexpr auto smooth = false;
        static constexpr auto basename = "MAE";

        ///
        /// \brief constructor
        ///
        explicit loss_mae_t(tensor_size_t dims);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx) const;
    };

    ///
    /// \brief hinge loss (linear SVM).
    ///
    class NANO_PUBLIC loss_hinge_t : public synthetic_sclass_t
    {
    public:

        static constexpr auto smooth = false;
        static constexpr auto basename = "Hinge";

        ///
        /// \brief constructor
        ///
        explicit loss_hinge_t(tensor_size_t dims);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx) const;
    };

    ///
    /// \brief logistic loss (binary classification).
    ///
    class NANO_PUBLIC loss_logistic_t : public synthetic_sclass_t
    {
    public:

        static constexpr auto smooth = true;
        static constexpr auto basename = "Logistic";

        ///
        /// \brief constructor
        ///
        explicit loss_logistic_t(tensor_size_t dims);

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx) const;
    };

    using function_enet_mae_t = function_enet_t<loss_mae_t>;
    using function_enet_mse_t = function_enet_t<loss_mse_t>;
    using function_enet_hinge_t = function_enet_t<loss_hinge_t>;
    using function_enet_logistic_t = function_enet_t<loss_logistic_t>;
}
