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
    subsample,          ///< (uniform) subsample the training samples (withput replacement)
    bootstrap,          ///< bootstrap the training samples (with replacement)
    wei_loss_bootstrap, ///< weighted (by the loss value) boostrapping of the training samples
    wei_grad_bootstrap, ///< weighted (by the loss gradient magnitudevalue) boostrapping of the training samples
};
} // namespace nano::gboost

NANO_MAKE_ENUM2(gboost::wscale_type, gboost, tboost)
NANO_MAKE_ENUM3(gboost::shrinkage_type, off, global, local)
NANO_MAKE_ENUM5(gboost::subsample_type, off, subsample, bootstrap, wei_loss_bootstrap, wei_grad_bootstrap)
