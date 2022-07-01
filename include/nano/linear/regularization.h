#pragma once

#include <nano/core/strutil.h>

namespace nano::linear
{
    ///
    /// \brief regularization methods for linear models.
    ///
    /// see "Regression Shrinkage and Selection via the lasso", by R. Tibshirani
    /// see "Empirical Bernstein Boosting", by Pannagadatta K. Shivaswamy & Tony Jebara
    /// see "Variance Penalizing AdaBoost", by Pannagadatta K. Shivaswamy & Tony Jebara
    ///
    enum class regularization_type
    {
        none,       ///< no regularization
        lasso,      ///< lasso
        ridge,      ///< ridge
        elasticnet, ///< elastic net
        variance    ///< variance of the loss values, like in VadaBoost
    };
} // namespace nano::linear

namespace nano
{
    template <>
    inline enum_map_t<linear::regularization_type> enum_string<linear::regularization_type>()
    {
        return {
            {      linear::regularization_type::none,       "none"},
            {     linear::regularization_type::lasso,      "lasso"},
            {     linear::regularization_type::ridge,      "ridge"},
            {linear::regularization_type::elasticnet, "elasticnet"},
            {  linear::regularization_type::variance,   "variance"}
        };
    }
} // namespace nano
