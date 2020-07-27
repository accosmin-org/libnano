#pragma once

#include <nano/string.h>

namespace nano
{
    ///
    /// \brief dataset splitting protocol.
    ///
    enum class protocol
    {
        train = 0,      ///< training
        valid,          ///< validation (for tuning hyper-parameters)
        test            ///< testing
    };

    template <>
    inline enum_map_t<protocol> enum_string<protocol>()
    {
        return
        {
            { protocol::train,    "train" },
            { protocol::valid,    "valid" },
            { protocol::test,     "test" }
        };
    }

    ///
    /// \brief execution policy.
    ///
    enum class execution
    {
        seq = 0,        ///< sequential: using only the thread
        par             ///< parallel: use all the available threads
    };

    ///
    /// \brief regularization methods.
    ///
    /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
    /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
    ///
    enum class regularization
    {
        none = 0,       ///< no regularization
        lasso,          ///< like in LASSO
        ridge,          ///< like in ridge regression, weight decay or Tikhonov regularization
        elastic,        ///< like in elastic net regularization
        variance        ///< like in VadaBoost or EBBoost
    };

    template <>
    inline enum_map_t<regularization> enum_string<regularization>()
    {
        return
        {
            { regularization::none,     "none" },
            { regularization::lasso,    "lasso" },
            { regularization::ridge,    "ridge" },
            { regularization::elastic,  "elastic" },
            { regularization::variance, "variance" }
        };
    }

    ///
    /// \brief input normalization (feature scaling) methods.
    ///
    enum class normalization
    {
        none = 0,       ///< no normalization, use the feature values as is
        mean,           ///< mean normalization: x = (x - mean(x)) / (max(x) - min(x))
        minmax,         ///< min-max normalization: x = (x - min(x)) / (max(x) - min(x))
        standard        ///< standardization with zero mean and unit variance: x = (x - min(x)) / stdev(x)
    };

    template <>
    inline enum_map_t<normalization> enum_string<normalization>()
    {
        return
        {
            { normalization::none,      "none" },
            { normalization::mean,      "mean" },
            { normalization::minmax,    "minmax" },
            { normalization::standard,  "standard" }
        };
    }

    ///
    /// \brief method to scale weak learners.
    ///
    enum class wscale : int32_t
    {
        gboost,         ///< use the same scaling factor for all samples (e.g. vanilla GradientBoosting)
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
        shuffle,        ///< impact on error rate by shuffling the feature values without retraining
        drop,           ///< impact on error rate by dropping the feature and retraining without it
    };

    template <>
    inline enum_map_t<importance> enum_string<importance>()
    {
        return
        {
            { importance::shuffle,  "shuffle" },
            { importance::drop,     "drop" },
        };
    }

    ///
    /// \brief hinge type (see MARS).
    ///
    /// see "Multivariate adaptive regression splines", by Jerome Friedman
    ///
    enum class hinge
    {
        left,       ///< beta * (threshold - x(feature))+       => zero on the right, linear on the left!
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
}
