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
} // namespace nano
