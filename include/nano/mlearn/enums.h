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
    enum class regularization
    {
        none = 0,       ///< no regularization
        lasso,          ///< like in LASSO
        ridge,          ///< like in ridge regression, weight decay or Tikhonov regularization
        elastic,        ///< like in elastic net regularization
        variance        ///< like in VadaBoost
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
}
