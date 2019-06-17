#pragma once

#include "regression.h"

namespace nano
{
        ///
        /// \brief square loss: l(x) = 1/2 * x^x.
        ///
        struct square_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return scalar_t(0.5) * (output - target).square().sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return output - target;
                }
        };

        using square_loss_t = regression_t<square_t>;
}
