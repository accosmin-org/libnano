#pragma once

#include <nano/tensor/eigen.h>

namespace nano::detail
{
///
/// \brief multi-class savage loss.
///
template <class terror>
struct savage_t : public terror
{
    static constexpr auto convex   = false;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "savage";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        const auto edges = (target * output).exp();

        return (1.0 / (1.0 + edges).square()).sum();
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        const auto edges = (target * output).exp();

        vgrad = -2.0 * target / (1.0 + edges).cube();
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        const auto edges = (target * output).exp();

        vhess = (-2.0 * (edges - 2.0 * edges.square()) / (1.0 + edges).square().square()).matrix().asDiagonal();
    }
};
} // namespace nano::detail
