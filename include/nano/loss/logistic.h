#pragma once

#include <nano/tensor/eigen.h>

namespace nano::detail
{
///
/// \brief multi-class logistic loss (see LogitBoost, logistic regression).
///
template <class terror>
struct logistic_t : public terror
{
    static constexpr auto convex   = true;
    static constexpr auto smooth   = true;
    static constexpr auto basename = "logistic";

    template <class tarray>
    requires is_eigen_v<tarray>
    static auto value(const tarray& target, const tarray& output)
    {
        scalar_t value = 0.0;
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            const auto x = -target(i) * output(i);
            value += (x < 1.0) ? std::log1p(std::exp(x)) : (x + std::log1p(std::exp(-x)));
        }
        return value;
    }

    template <class tarray, class tgarray>
    requires(is_eigen_v<tarray> && is_eigen_v<tgarray>)
    static void vgrad(const tarray& target, const tarray& output, tgarray vgrad)
    {
        for (tensor_size_t i = 0, size = target.size(); i < size; ++i)
        {
            const auto x = -target(i) * output(i);
            const auto g = (x < 1.0) ? (std::exp(x) / (1.0 + std::exp(x))) : (1.0 / (1.0 + std::exp(-x)));
            vgrad(i)     = -target(i) * g;
        }
    }

    template <class tarray, class thmatrix>
    requires(is_eigen_v<tarray> && is_eigen_v<thmatrix>)
    static void vhess(const tarray& target, const tarray& output, thmatrix vhess)
    {
        vhess.full(0.0);
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
