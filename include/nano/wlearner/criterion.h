#pragma once

#include <nano/arch.h>
#include <nano/core/enumutil.h>
#include <nano/core/strutil.h>

namespace nano
{
///
/// \brief criteria to select weak learner, trading between fitting and complexity.
///
enum class wlearner_criterion
{
    rss,  ///< residual sum of squares
    aic,  ///< AIC
    aicc, ///< AICc
    bic,  ///< BIC
};
NANO_MAKE_ENUM4(wlearner_criterion, rss, aic, aicc, bic)
} // namespace nano

namespace nano::wlearner
{
NANO_PUBLIC double make_score(wlearner_criterion, double rss, int64_t k, int64_t n);
} // namespace nano::wlearner
