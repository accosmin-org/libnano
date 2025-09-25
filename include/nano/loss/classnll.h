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
        tensor_size_t imax = 0;
        const auto    omax = output.maxCoeff(&imax);

        scalar_t value = std::numeric_limits<scalar_t>::epsilon();
        scalar_t posum = 0;
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
        tensor_size_t imax = 0;
        const auto    omax = output.maxCoeff(&imax);

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
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess.fill(0.0);
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            const auto x = -target(i) * output(i);
            const auto h = (x < 1.0) ? (std::exp(x) * (1.0 - std::exp(x)) / square(1.0 + std::exp(x)))
                                     : ((std::exp(-x) - 1) / square(1.0 + std::exp(-x)));
            vhess(i, i)  = target(i) * target(i) * h;
        }
    }
};
} // namespace nano::detail
