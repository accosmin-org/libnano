#pragma once

#include <nano/core/enumutil.h>
#include <nano/core/strutil.h>

namespace nano
{
///
/// \brief machine learning task type.
///
enum class task_type : int32_t
{
    regression = 0,  ///< regression
    sclassification, ///< single-label classification
    mclassification, ///< multi-label classification
    unsupervised,    ///< unsupervised
};
NANO_MAKE_ENUM4(task_type, regression, sclassification, mclassification, unsupervised)
} // namespace nano
