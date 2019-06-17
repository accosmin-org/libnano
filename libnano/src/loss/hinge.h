#pragma once

#include "classification.h"

namespace nano
{
        ///
        /// \brief multi-class hinge loss: sum(max(0, 1 - target_k * output_k), k).
        ///
        struct hinge_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return (1 - target * output).max(0).sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return  -target * (1 - target * output).sign();
                }
        };

        using mhinge_loss_t = mclassification_t<hinge_t>;
        using shinge_loss_t = sclassification_t<hinge_t>;
}
