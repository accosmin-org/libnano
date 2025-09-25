#pragma once

#include <nano/tensor/eigen.h>

namespace nano::detail
{
///
/// \brief class negative log-likelihood loss (also called cross-entropy loss).
///
template <class terror>
struct classnll_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "classnll";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        auto       imax = tensor_size_t{0};
        const auto omax = output.maxCoeff(&imax);

        return std::log((output - omax).exp().sum()) - 0.5 * ((1.0 + target) * output).sum() + omax;
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        auto       imax = tensor_size_t{0};
        const auto omax = output.maxCoeff(&imax);

        vgrad = (output - omax).exp();
        vgrad /= vgrad.sum();
        vgrad -= 0.5 * (1.0 + target);
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess([[maybe_unused]] const tarray& target, const tarray& output, thmatrix vhess)
    {
        auto       imax = tensor_size_t{0};
        const auto omax = output.maxCoeff(&imax);

        const auto exps = (output - omax).exp();
        const auto coef = exps / exps.sum();

        vhess = -coef.matrix() * coef.matrix().transpose();
        vhess.diagonal().array() += coef;
    }
};
} // namespace nano::detail
