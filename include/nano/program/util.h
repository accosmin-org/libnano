#pragma once

#include <nano/tensor.h>

namespace nano::program
{
///
/// \brief return true if the given matrix is positive semi-definite.
///
NANO_PUBLIC bool is_psd(matrix_cmap_t);

///
/// \brief return true if the equality constraint `Ax = b` is not full row rank.
///
/// in this case the constraints are transformed in-place to obtain row-independant linear constraints
///     by performing an appropriate matrix decomposition.
///
NANO_PUBLIC bool reduce(matrix_t& A, vector_t& b);
} // namespace nano::program
