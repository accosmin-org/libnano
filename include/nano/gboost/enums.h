#pragma once

#include <nano/core/strutil.h>

namespace nano::gboost
{
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
/// \brief toggle shrinkage of the fitted weak learners at each boosting round.
///
/// see "Stochastic Gradient Boosting", by Jerome Friedman
///
enum class shrinkage_type : int32_t
{
    off,    ///< no shrinkage
    global, ///< same value for all boosting rounds (see reference)
    local,  ///< different values per boosting round
};

///
/// \brief toggle sub-sampling of the training samples at each boosting round.
///
/// see "Stochastic Gradient Boosting", by Jerome Friedman
///
enum class subsample_type : int32_t
{
    off,                ///< always use all available training samples
    bootstrap,          ///< bootstrap the training samples
    wei_loss_bootstrap, ///< weighted (by the loss value) boostrapping of the training samples
    wei_grad_bootstrap, ///< weighted (by the loss gradient magnitudevalue) boostrapping of the training samples
};
} // namespace nano::gboost

namespace nano
{
template <>
inline enum_map_t<gboost::wscale_type> enum_string<gboost::wscale_type>()
{
    return {
        {gboost::wscale_type::gboost, "gboost"},
        {gboost::wscale_type::tboost, "tboost"},
    };
}

template <>
inline enum_map_t<gboost::shrinkage_type> enum_string<gboost::shrinkage_type>()
{
    return {
        {   gboost::shrinkage_type::off,    "off"},
        {gboost::shrinkage_type::global, "global"},
        { gboost::shrinkage_type::local,  "local"},
    };
}

template <>
inline enum_map_t<gboost::subsample_type> enum_string<gboost::subsample_type>()
{
    return {
        {               gboost::subsample_type::off,                "off"},
        {         gboost::subsample_type::bootstrap,          "bootstrap"},
        {gboost::subsample_type::wei_loss_bootstrap, "wei_loss_bootstrap"},
        {gboost::subsample_type::wei_grad_bootstrap, "wei_grad_bootstrap"},
    };
}
} // namespace nano
