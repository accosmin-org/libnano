#pragma once

#include "classification.h"

namespace nano
{
        ///
        /// \brief class negative log-likelihood loss (also called cross-entropy loss).
        ///
        struct classnll_t
        {
                template <typename tarray>
                static auto value(const tarray& target, const tarray& output)
                {
                        return  std::log(output.exp().sum()) -
                                std::log(((1 + target) * output.exp()).sum() / 2);
                }

                template <typename tarray>
                static auto vgrad(const tarray& target, const tarray& output)
                {
                        return  output.exp() / output.exp().sum() -
                                (1 + target) * output.exp() / ((1 + target) * output.exp()).sum();
                }
        };

        using sclassnll_loss_t = sclassification_t<classnll_t>;
        using mclassnll_loss_t = mclassification_t<classnll_t>;
}
