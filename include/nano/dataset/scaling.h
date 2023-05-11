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
} // namespace nano

NANO_MAKE_ENUM4(scaling_type, none, mean, minmax, standard)
