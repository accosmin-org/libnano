#include <nano/core/stats.h>
#include <nano/wlearner/criterion.h>

using namespace nano;

double nano::wlearner::make_score(const wlearner_criterion criterion, double rss, const int64_t k, const int64_t n)
{
    rss = std::max(rss, std::numeric_limits<double>::epsilon() * 1e+3);

    switch (criterion)
    {
    case wlearner_criterion::aic: return AIC(rss, k, n);
    case wlearner_criterion::aicc: return AICc(rss, k, n);
    case wlearner_criterion::bic: return BIC(rss, k, n);
    default: return rss;
    }
}
