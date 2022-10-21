#pragma once

#include <nano/core/strutil.h>

namespace nano
{
    ///
    /// \brief method to scale weak learners.
    ///
    enum class wscale_type : int32_t
    {
        gboost = 0, ///< use the same scaling factor for all samples (e.g. vanilla GradientBoosting)
        tboost,     ///< use a potentially different scaling factor for each split (e.g. see TreeBoost variation)
    };

    template <>
    inline enum_map_t<wscale_type> enum_string<wscale_type>()
    {
        return {
            {wscale_type::gboost, "gboost"},
            {wscale_type::tboost, "tboost"}
        };
    }

    ///
    /// \brief method to estimate the importance of a feature.
    ///
    enum class importance_type : int32_t
    {
        shuffle = 0, ///< impact on the error rate by shuffling the feature values across samples without retraining
        dropcol,     ///< impact on the error rate by dropping the feature (aka column) and retraining without it
    };

    template <>
    inline enum_map_t<importance_type> enum_string<importance_type>()
    {
        return {
            {importance_type::shuffle, "shuffle"},
            {importance_type::dropcol, "dropcol"},
        };
    }

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

    template <>
    inline enum_map_t<ensemble_type> enum_string<ensemble_type>()
    {
        return {
            { ensemble_type::bumping,  "bumping"},
            {ensemble_type::stacking, "stacking"},
            { ensemble_type::bagging,  "average"},
            {  ensemble_type::median,   "median"},
        };
    }
} // namespace nano
