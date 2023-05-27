#pragma once

#include <nano/core/enumutil.h>
#include <nano/core/strutil.h>

namespace nano
{
///
/// \brief regularization methods for linear models.
///
/// see "Regression Shrinkage and Selection via the lasso", by R. Tibshirani
/// see "Regularization and variable selection via the elastic net", by H. Zou, T. Hastie
///
enum class linear_regularization
{
    none,       ///< no regularization
    lasso,      ///< lasso
    ridge,      ///< ridge
    elasticnet, ///< elastic net
};
NANO_MAKE_ENUM4(linear_regularization, none, lasso, ridge, elasticnet)
} // namespace nano
