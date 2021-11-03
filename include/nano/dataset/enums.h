#pragma once

#include <nano/core/strutil.h>

namespace nano
{
    ///
    /// \brief machine learning task type.
    ///
    enum class task_type : int32_t
    {
        regression = 0,     ///< regression
        sclassification,    ///< single-label classification
        mclassification,    ///< multi-label classification
        unsupervised,       ///< unsupervised
    };

    template <>
    inline enum_map_t<task_type> enum_string<task_type>()
    {
        return
        {
            { task_type::regression,        "regression" },
            { task_type::sclassification,   "s-classification" },
            { task_type::mclassification,   "m-classification" },
            { task_type::unsupervised,      "unsupervised" },
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, task_type value)
    {
        return stream << scat(value);
    }

    ///
    /// \brief input or target feature type.
    ///
    enum class feature_type : int32_t
    {
        // continuous features
        int8 = 0,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,

        // discrete features
        sclass,         ///< categorical feature (single-label - one value possible out of a fixed set)
        mclass,         ///< categorical feature (mulit-label - a subset of values possible out of a fixed set)
    };

    template <>
    inline enum_map_t<feature_type> enum_string<feature_type>()
    {
        return
        {
            { feature_type::int8,       "int8" },
            { feature_type::int16,      "int16" },
            { feature_type::int32,      "int32" },
            { feature_type::int64,      "int64" },
            { feature_type::uint8,      "uint8" },
            { feature_type::uint16,     "uint16" },
            { feature_type::uint32,     "uint32" },
            { feature_type::uint64,     "uint64" },
            { feature_type::float32,    "float32" },
            { feature_type::float64,    "float64" },
            { feature_type::sclass,     "sclass" },
            { feature_type::mclass,     "mclass" },
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, feature_type value)
    {
        return stream << scat(value);
    }

    ///
    /// \brief scaling methods for flatten input features and scalar targets.
    ///
    enum class scaling_type : int32_t
    {
        none = 0,       ///< no normalization, use the feature values as is
        mean,           ///< mean normalization: x = (x - mean(x)) / (max(x) - min(x))
        minmax,         ///< min-max normalization: x = (x - min(x)) / (max(x) - min(x))
        standard        ///< standardization with zero mean and unit variance: x = (x - mean(x)) / stdev(x)
    };

    template <>
    inline enum_map_t<scaling_type> enum_string<scaling_type>()
    {
        return
        {
            { scaling_type::none,    "none" },
            { scaling_type::mean,    "mean" },
            { scaling_type::minmax,  "minmax" },
            { scaling_type::standard,"standard" }
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, scaling_type value)
    {
        return stream << scat(value);
    }
}
