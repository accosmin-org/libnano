#pragma once

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

template <>
inline enum_map_t<task_type> enum_string()
{
    return {
        {     task_type::regression,      "regression"},
        {task_type::sclassification, "sclassification"},
        {task_type::mclassification, "mclassification"},
        {   task_type::unsupervised,    "unsupervised"}
    };
}
} // namespace nano
