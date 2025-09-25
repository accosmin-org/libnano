#pragma once

#include <nano/tensor/eigen.h>

namespace nano::detail
{
///
/// \brief robust-to-noise Cauchy loss.
///
template <class terror>
struct cauchy_t : public terror
{
    static constexpr auto convex   = false;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "cauchy";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        return 0.5 * ((target - output).square() + 1).log().sum();
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        vgrad = (output - target) / (1 + (output - target).square());
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess = ((output - target) / (1 + (output - target).square()).square()).matrix().asDiagonal();
    }
};
} // namespace nano::detail
