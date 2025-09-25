#pragma once

#include <nano/tensor/eigen.h>

namespace nano::detail
{
///
/// \brief mean absolute error (MAE).
///
template <class terror>
struct mae_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = false;
    static constexpr auto basename = "mae";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return (output - target).abs().sum();
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = (output - target).sign();
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess([[maybe_unused]] const tarray& target, [[maybe_unused]] const tarray& output, thmatrix vhess)
    {
        assert(false);
        vhess.fill(0.0);
    }
};
} // namespace nano::detail
