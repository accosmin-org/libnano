#pragma once

#include <nano/core/strutil.h>

namespace nano
{
    ///
    /// \brief scaling methods for flattening input features and scalar targets.
    ///
    enum class scaling_type : int32_t
    {
        none = 0, ///< no normalization, use the feature values as is
        mean,     ///< mean normalization: x = (x - mean(x)) / (max(x) - min(x))
        minmax,   ///< min-max normalization: x = (x - min(x)) / (max(x) - min(x))
        standard  ///< standardization with zero mean and unit variance: x = (x - mean(x)) / stdev(x)
    };

    template <>
    inline enum_map_t<scaling_type> enum_string<scaling_type>()
    {
        return {
            {    scaling_type::none,     "none"},
            {    scaling_type::mean,     "mean"},
            {  scaling_type::minmax,   "minmax"},
            {scaling_type::standard, "standard"}
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, scaling_type value) { return stream << scat(value); }
} // namespace nano
