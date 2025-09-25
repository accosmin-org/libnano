#pragma once

#include <nano/tensor/eigen.h>

namespace nano::detail
{
///
/// \brief multi-class tangent loss.
///
template <class terror>
struct tangent_t : public terror
{
    static constexpr auto convex   = false;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "tangent";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        const auto atan = 2.0 * (target * output).atan() - 1.0;

        return atan.square().sum();
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        const auto atan = 2.0 * (target * output).atan() - 1.0;
        const auto gdiv = 1.0 + (target * output).square();

        vgrad = 4.0 * target * atan / gdiv;
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        const auto atan = 2.0 * (target * output).atan() - 1.0;
        const auto gdiv = 1.0 + (target * output).square();

        vhess = (8.0 * target * target * (1.0 - output * target * atan) / gdiv.square()).matrix().asDiagonal();
    }
};
} // namespace nano::detail
