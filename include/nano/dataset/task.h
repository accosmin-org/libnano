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
    inline enum_map_t<task_type> enum_string<task_type>()
    {
        return {
            {     task_type::regression,       "regression"},
            {task_type::sclassification, "s-classification"},
            {task_type::mclassification, "m-classification"},
            {   task_type::unsupervised,     "unsupervised"},
        };
    }

    inline std::ostream& operator<<(std::ostream& stream, task_type value) { return stream << scat(value); }
} // namespace nano
