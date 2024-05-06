#pragma once

#include <nano/core/strutil.h>

namespace nano
{
///
/// \brief regularization methods for linear models.
///
/// see "Regression Shrinkage and Selection via the lasso", by R. Tibshirani
/// see "Regularization and variable selection via the elastic net", by H. Zou, T. Hastie
///
enum class linear_regularization : uint8_t
{
    none,       ///< no regularization
    lasso,      ///< lasso
    ridge,      ///< ridge
    elasticnet, ///< elastic net
};

template <>
inline enum_map_t<linear_regularization> enum_string()
{
    return {
        {      linear_regularization::none,       "none"},
        {     linear_regularization::lasso,      "lasso"},
        {     linear_regularization::ridge,      "ridge"},
        {linear_regularization::elasticnet, "elasticnet"}
    };
}
} // namespace nano
