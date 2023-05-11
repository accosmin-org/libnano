#pragma once

#include <nano/core/strutil.h>

namespace nano
{
///
/// \brief method to estimate the importance of a feature.
///
enum class importance_type : int32_t
{
    shuffle = 0, ///< impact on the error rate by shuffling the feature values across samples without retraining
    dropcol,     ///< impact on the error rate by dropping the feature (aka column) and retraining without it
};

///
/// \brief methods to combine the predictions of different models trained on different folds.
///
/// see "Bagging Predictors", by Leo Breiman
/// see "Stacked Regressions", by Leo Breiman
/// see "Model search and inference by bootstrap bumping", by R. Tibshirani and K. Knight
/// see "Combining estimates in regression and classification", by M. LeBlanc and R. Tibshirani
///
enum class ensemble_type : int32_t
{
    bumping = 0, ///< see bumping
    stacking,    ///< see stacking
    bagging,     ///< see bagging
    median,      ///< see bagging, but output the median per sample of the models' predictions
};
} // namespace nano

NANO_MAKE_ENUM2(importance_type, shuffle, dropcol)
NANO_MAKE_ENUM4(ensemble_type, bumping, stacking, bagging, median)
