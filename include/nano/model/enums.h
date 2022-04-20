#pragma once

#include <nano/core/strutil.h>

namespace nano
{
    ///
    /// \brief method to scale weak learners.
    ///
    enum class wscale : int32_t
    {
        gboost = 0,     ///< use the same scaling factor for all samples (e.g. vanilla GradientBoosting)
        tboost,         ///< use a potentially different scaling factor for each split (e.g. see TreeBoost variation)
    };

    template <>
    inline enum_map_t<wscale> enum_string<wscale>()
    {
        return
        {
            { wscale::gboost,       "gboost" },
            { wscale::tboost,       "tboost" }
        };
    }

    ///
    /// \brief method to estimate the importance of a feature.
    ///
    enum class importance : int32_t
    {
        shuffle = 0,    ///< impact on the error rate by shuffling the feature values across samples without retraining
        dropcol,        ///< impact on the error rate by dropping the feature (aka column) and retraining without it
    };

    template <>
    inline enum_map_t<importance> enum_string<importance>()
    {
        return
        {
            { importance::shuffle,  "shuffle" },
            { importance::dropcol,  "dropcol" },
        };
    }

    ///
    /// \brief hinge type (see MARS).
    ///
    /// see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    enum class hinge : int32_t
    {
        left = 0,   ///< beta * (threshold - x(feature))+       => zero on the right, linear on the left!
        right,      ///< beta * (x(feature) - threshold)+       => zero on the left, linear on the right!
    };

    template <>
    inline enum_map_t<hinge> enum_string<hinge>()
    {
        return
        {
            { hinge::left,          "left" },
            { hinge::right,         "right" },
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
    enum class ensemble : int32_t
    {
        bumping = 0,///< see bumping
        stacking,   ///< see stacking
        bagging,    ///< see bagging
        median,     ///< see bagging, but output the median per sample of the models' predictions
    };

    template <>
    inline enum_map_t<ensemble> enum_string<ensemble>()
    {
        return
        {
            { ensemble::bumping,    "bumping" },
            { ensemble::stacking,   "stacking" },
            { ensemble::bagging,    "average" },
            { ensemble::median,     "median" },
        };
    }
}
