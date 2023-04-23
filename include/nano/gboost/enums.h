#pragma once

#include <nano/core/strutil.h>

namespace nano::gboost
{
///
/// \brief regularization methods for gradient boosting-based models.
///
/// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
/// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
///
enum class regularization_type
{
    none,    ///< no regularization
    variance ///< variance of the loss values, like in VadaBoost
};

///
/// \brief method to scale weak learners.
///
/// see "Greedy Function Approximation: A Gradient Boosting Machine", by Jerome Friedman
///
enum class wscale_type : int32_t
{
    gboost = 0, ///< use the same scaling factor for all samples (e.g. vanilla GradientBoosting)
    tboost,     ///< use a potentially different scaling factor for each split (e.g. see TreeBoost variation)
};

///
/// \brief toggle subsampling of the training samples at each boosting round.
///
/// see "Stochastic Gradient Boosting", by Jerome Friedman
///
///
enum class subsample_type : int32_t
{
    on,
    off
};

///
/// \brief toggle shrinkage of the fitted weak learners at each boosting round.
///
/// see "Stochastic Gradient Boosting", by Jerome Friedman
///
enum class shrinkage_type : int32_t
{
    on,
    off
};

///
/// \brief toggle bootstraping of the training samples at each boosting round (as an alternative to subsampling).
///
enum class bootstrap_type : int32_t
{
    on,
    off
};
} // namespace nano::gboost

namespace nano
{
template <>
inline enum_map_t<gboost::regularization_type> enum_string<gboost::regularization_type>()
{
    return {
        {    gboost::regularization_type::none,     "none"},
        {gboost::regularization_type::variance, "variance"}
    };
}

template <>
inline enum_map_t<gboost::wscale_type> enum_string<gboost::wscale_type>()
{
    return {
        {gboost::wscale_type::gboost, "gboost"},
        {gboost::wscale_type::tboost, "tboost"}
    };
}

template <>
inline enum_map_t<gboost::subsample_type> enum_string<gboost::subsample_type>()
{
    return {
        { gboost::subsample_type::on,  "on"},
        {gboost::subsample_type::off, "off"}
    };
}

template <>
inline enum_map_t<gboost::shrinkage_type> enum_string<gboost::shrinkage_type>()
{
    return {
        { gboost::shrinkage_type::on,  "on"},
        {gboost::shrinkage_type::off, "off"}
    };
}

template <>
inline enum_map_t<gboost::bootstrap_type> enum_string<gboost::bootstrap_type>()
{
    return {
        { gboost::bootstrap_type::on,  "on"},
        {gboost::bootstrap_type::off, "off"}
    };
}
} // namespace nano
