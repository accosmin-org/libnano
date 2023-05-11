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

NANO_MAKE_ENUM4(linear::regularization_type, none, lasso, ridge, elasticnet)
