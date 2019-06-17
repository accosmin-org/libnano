#pragma once

#include "classification.h"

namespace nano
{
        ///
        /// \brief multi-class logistic loss: sum(log(1 + exp(-target_k * output_k)), k).
        ///
        struct logistic_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return  (1 + (-target * output).exp()).log().sum();
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return  -target * (-target * output).exp() /
                                (1 + (-target * output).exp());
                }
        };

        using mlogistic_loss_t = mclassification_t<logistic_t>;
        using slogistic_loss_t = sclassification_t<logistic_t>;
}
