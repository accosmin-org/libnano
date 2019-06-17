#pragma once

#include "regression.h"

namespace nano
{
        ///
        /// \brief robust-to-noise Cauchy loss: 1/2 * log(1 + x^2).
        ///
        struct cauchy_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return scalar_t(0.5) * ((target - output).square() + 1).log().sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return (output - target) / (1 + (output - target).square());
                }
        };

        using cauchy_loss_t = regression_t<cauchy_t>;
}
