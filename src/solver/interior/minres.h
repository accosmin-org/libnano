#pragma once

#include <nano/tensor.h>

namespace nano
{
///
/// \brief MINRES algorithm to solve the system `A * x = b`, where A is a symmetric matrix.
///
/// see "Solution of sparse indefinite systems of linear equations", by C. C. Paige, M. A. Saunders (1975).
///
bool MINRES(const matrix_t& A, const vector_t& b, vector_t& x, tensor_size_t max_iters = 1000,
            scalar_t tolerance = 1e-15);
} // namespace nano
