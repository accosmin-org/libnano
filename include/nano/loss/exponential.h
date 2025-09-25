#pragma once

#include <nano/tensor/eigen.h>

namespace nano::detail
{
///
/// \brief multi-class exponential loss (see AdaBoost).
///
template <class terror>
struct exponential_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "exponential";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return (-target * output).exp().sum();
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = -target * (-target * output).exp();
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess = (-target * output).exp().matrix().asDiagonal();
    }
};
} // namespace nano::detail
