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

        auto value = std::numeric_limits<scalar_t>::epsilon();
        auto posum = 0.0;
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            value += std::exp(output(i) - omax);
            if (is_pos_target(target(i)))
            {
                posum += output(i);
            }
        }
        return std::log(value) - posum + omax;
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        auto       imax = tensor_size_t{0};
        const auto omax = output.maxCoeff(&imax);

        scalar_t value = 0;
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            value += (vgrad(i) = std::exp(output(i) - omax));
        }
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            vgrad(i) /= value;
            if (is_pos_target(target(i)))
            {
                vgrad(i) -= 1.0;
            }
        }
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess([[maybe_unused]] const tarray& target, const tarray& output, thmatrix vhess)
    {
        auto       imax = tensor_size_t{0};
        const auto omax = output.maxCoeff(&imax);

        const auto exps = (output - omax).exp();
        const auto esum = exps.sum();

        const auto coef = exps / esum;

        vhess = -coef.matrix() * coef.matrix().transpose();
        vhess.diagonal().array() += coef;
    }
};
} // namespace nano::detail
