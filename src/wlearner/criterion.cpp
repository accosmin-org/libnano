#include <nano/core/stats.h>
#include <nano/wlearner/criterion.h>

using namespace nano;

double nano::wlearner::make_score(const criterion_type criterion, double rss, const int64_t k, const int64_t n)
{
    rss = std::max(rss, std::numeric_limits<double>::epsilon());

    switch (criterion)
    {
    case criterion_type::aic: return AIC(rss, k, n);
    case criterion_type::aicc: return AICc(rss, k, n);
    case criterion_type::bic: return BIC(rss, k, n);
    default: return rss;
    }
}
