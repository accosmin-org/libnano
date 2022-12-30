#pragma once

#include <nano/core/strutil.h>

namespace nano::gboost
{
    ///
    /// \brief regularization methods for linear models.
    ///
    /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
    /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
    ///
    enum class regularization_type
    {
        none,    ///< no regularization
        variance ///< variance of the loss values, like in VadaBoost
    };

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
    inline enum_map_t<linear::regularization_type> enum_string<linear::regularization_type>()
    {
        return {
            {    linear::regularization_type::none,     "none"},
            {linear::regularization_type::variance, "variance"}
        };
    }

    template <>
    inline enum_map_t<gboost::wscale_type> enum_string<gboost::wscale_type>()
    {
        return {
            {gboost::wscale_type::gboost, "gboost"},
            {gboost::wscale_type::tboost, "tboost"}
        };
    }
} // namespace nano
