#pragma once

#include "classification.h"

namespace nano
{
        ///
        /// \brief multi-class exponential loss: sum(exp(-target_k * output_k), k).
        ///
        struct exponential_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return (-target * output).exp().sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return -target * (-target * output).exp();
                }
        };

        using mexponential_loss_t = mclassification_t<exponential_t>;
        using sexponential_loss_t = sclassification_t<exponential_t>;
}
