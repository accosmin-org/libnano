#pragma once

#include <nano/tensor/eigen.h>

namespace nano::detail
{
///
/// \brief mean squared error (MSE).
///
template <class terror>
struct mse_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "mse";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return 0.5 * (output - target).square().sum();
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = output - target;
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess([[maybe_unused]] const tarray& target, [[maybe_unused]] const tarray& output, thmatrix vhess)
    {
        vhess = matrix_t::identity(vhess.rows(), vhess.cols());
    }
};
} // namespace nano::detail
