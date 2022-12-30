#pragma once

#include <nano/core/strutil.h>

namespace nano::gboost
{
    ///
    /// \brief method to scale weak learners.
    ///
    enum class wscale_type : int32_t
    {
        gboost = 0, ///< use the same scaling factor for all samples (e.g. vanilla GradientBoosting)
        tboost,     ///< use a potentially different scaling factor for each split (e.g. see TreeBoost variation)
    };
} // namespace nano::gboost

namespace nano
{
    template <>
    inline enum_map_t<gboost::wscale_type> enum_string<gboost::wscale_type>()
    {
        return {
            {gboost::wscale_type::gboost, "gboost"},
            {gboost::wscale_type::tboost, "tboost"}
        };
    }
} // namespace nano
