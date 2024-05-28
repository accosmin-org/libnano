#pragma once

#include <nano/enum.h>

namespace nano
{
///
/// \brief method to scale weak learners.
///
/// see "Greedy Function Approximation: A Gradient Boosting Machine", by Jerome Friedman
///
enum class gboost_wscale : uint8_t
{
    gboost = 0, ///< use the same scaling factor for all samples (e.g. vanilla GradientBoosting)
    tboost,     ///< use a potentially different scaling factor for each split (e.g. see TreeBoost variation)
};

template <>
inline enum_map_t<gboost_wscale> enum_string()
{
    return {
        {gboost_wscale::gboost, "gboost"},
        {gboost_wscale::tboost, "tboost"}
    };
}

///
/// \brief toggle shrinkage of the fitted weak learners at each boosting round.
///
/// see "Stochastic Gradient Boosting", by Jerome Friedman
///
enum class gboost_shrinkage : uint8_t
{
    off,    ///< no shrinkage
    global, ///< same value for all boosting rounds (see reference)
    local,  ///< different values per boosting round
};

template <>
inline enum_map_t<gboost_shrinkage> enum_string()
{
    return {
        {   gboost_shrinkage::off,    "off"},
        {gboost_shrinkage::global, "global"},
        { gboost_shrinkage::local,  "local"}
    };
}

///
/// \brief toggle sub-sampling of the training samples at each boosting round.
///
/// see "Stochastic Gradient Boosting", by Jerome Friedman
///
enum class gboost_subsample : uint8_t
{
    off,                ///< always use all available training samples
    subsample,          ///< (uniform) subsample the training samples (withput replacement)
    bootstrap,          ///< bootstrap the training samples (with replacement)
    wei_loss_bootstrap, ///< weighted (by the loss value) boostrapping of the training samples
    wei_grad_bootstrap, ///< weighted (by the loss gradient magnitudevalue) boostrapping of the training samples
};

template <>
inline enum_map_t<gboost_subsample> enum_string()
{
    return {
        {               gboost_subsample::off,                "off"},
        {         gboost_subsample::subsample,          "subsample"},
        {         gboost_subsample::bootstrap,          "bootstrap"},
        {gboost_subsample::wei_loss_bootstrap, "wei_loss_bootstrap"},
        {gboost_subsample::wei_grad_bootstrap, "wei_grad_bootstrap"}
    };
}
} // namespace nano
