#pragma once

#include <nano/arch.h>
#include <nano/core/strutil.h>

namespace nano::wlearner
{
///
/// \brief criteria to select weak learner, trading between fitting and complexity.
///
enum class criterion_type
{
    rss,  ///< residual sum of squares
    aic,  ///< AIC
    aicc, ///< AICc
    bic,  ///< BIC
};

NANO_PUBLIC double make_score(criterion_type, double rss, int64_t k, int64_t n);
} // namespace nano::wlearner

NANO_MAKE_ENUM4(wlearner::criterion_type, rss, aic, aicc, bic)
