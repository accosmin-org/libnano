#pragma once

#include <nano/core/strutil.h>

namespace nano::linear
{
///
/// \brief regularization methods for linear models.
///
/// see "Regression Shrinkage and Selection via the lasso", by R. Tibshirani
/// see "Regularization and variable selection via the elastic net", by H. Zou, T. Hastie
///
enum class regularization_type
{
    none,       ///< no regularization
    lasso,      ///< lasso
    ridge,      ///< ridge
    elasticnet, ///< elastic net
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
        {linear::regularization_type::elasticnet, "elasticnet"}
    };
}
} // namespace nano
