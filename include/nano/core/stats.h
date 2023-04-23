#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <numeric>
#include <ostream>
#include <vector>

namespace nano
{
namespace detail
{
template <typename titerator, typename toperator>
auto percentile(titerator begin, titerator end, const double percentage, const toperator& from_position) noexcept
{
    assert(percentage >= 0.0 && percentage <= 100.0);

    const auto   size     = std::distance(begin, end);
    const double position = percentage * static_cast<double>(size - 1) / 100.0;

    const auto lpos = static_cast<decltype(size)>(std::floor(position));
    const auto rpos = static_cast<decltype(size)>(std::ceil(position));

    if (lpos == rpos)
    {
        return from_position(lpos);
    }
    else
    {
        const auto lvalue = from_position(lpos);
        const auto rvalue = from_position(rpos);
        return (lvalue + rvalue) / 2;
    }
}
} // namespace detail

///
/// \brief returns the percentile value from a potentially not sorted list of values.
///
template <typename titerator>
auto percentile(titerator begin, titerator end, const double percentage) noexcept
{
    const auto from_position = [begin = begin, end = end](auto pos)
    {
        auto middle = begin;
        std::advance(middle, pos);
        std::nth_element(begin, middle, end);
        return static_cast<double>(*middle);
    };

    return detail::percentile(begin, end, percentage, from_position);
}

///
/// \brief returns the percentile value from a sorted list of values.
///
template <typename titerator>
auto percentile_sorted(titerator begin, titerator end, const double percentage) noexcept
{
    assert(std::is_sorted(begin, end));

    const auto from_position = [begin = begin](auto pos)
    {
        auto middle = begin;
        std::advance(middle, pos);
        return static_cast<double>(*middle);
    };

    return detail::percentile(begin, end, percentage, from_position);
}

///
/// \brief returns the median value from a potentially not sorted list of values.
///
template <typename titerator>
auto median(titerator begin, titerator end) noexcept
{
    return percentile(begin, end, 50);
}

///
/// \brief returns the median value from a sorted list of values.
///
template <typename titerator>
auto median_sorted(titerator begin, titerator end) noexcept
{
    return percentile_sorted(begin, end, 50);
}

///
/// \brief returns the Akaike information criterion (AIC) given the residual sum of squares (RSS),
///     the number of effective parameters (k) and the number of samples (n).
///
/// NB: models the trade-off between the goodness of fit and the model simplificity.
///
/// see (1) - "A new look at the statistical model identification", by H. Akaike, 1974.
///
inline double AIC(const double RSS, const int64_t k, const int64_t n)
{
    assert(n > 0);
    assert(k > 0);
    assert(RSS > 0.0);

    const auto dk = static_cast<double>(k);
    const auto dn = static_cast<double>(n);

    return 2.0 * dk + dn * std::log(RSS) - dn * std::log(dn);
}

///
/// \brief returns the corrected Akaike information criterion (AICc) given the residual sum of squares (RSS),
///     the number of effective parameters (k) and the number of samples (n).
///
/// NB: models the trade-off between the goodness of fit and the model simplificity.
/// NB: the correction (compared to AIC) is effective to small sample size.
///
/// see (1) - "Regression and time series model selection in small samples", by C. M. Hurvich and C. L. Tsai, 1989.
///
inline double AICc(const double RSS, const int64_t k, const int64_t n)
{
    const auto dk = static_cast<double>(k);
    const auto dn = static_cast<double>(n);

    return AIC(RSS, k, n) + 2.0 * (dk * dk + dk) / (dn - dk - 1.0);
}

///
/// \brief returns the Bayesian information criterion (BIC) given the residual sum of squares (RSS),
///     the number of effective parameters (k) and the number of samples (n).
///
/// NB: models the trade-off between the goodness of fit and the model simplificity.
///
/// see (1) - "Estimating the dimension of a model", by G. Schwarz, 1978.
///
inline double BIC(const double RSS, const int64_t k, const int64_t n)
{
    assert(n > 0);
    assert(k > 0);
    assert(RSS > 0.0);

    const auto dk = static_cast<double>(k);
    const auto dn = static_cast<double>(n);

    return dk * std::log(dn) + dn * std::log(RSS / dn);
}
} // namespace nano
