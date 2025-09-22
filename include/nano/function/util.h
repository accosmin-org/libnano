#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief return the minimum difference between the analytical gradient and the gradient approximated
///     using central finite difference approximation.
///
/// see (1) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
/// NB: if significantly bigger than zero, then the (sub-)gradient is not correctly computed.
///
NANO_PUBLIC scalar_t grad_accuracy(const function_t&, const vector_t& x, scalar_t early_stopping_epsilon = 1e-10);

///
/// \brief return the minimum difference between the analytical hessian and the hessian approximated
///     using central finite difference approximation.
///
/// see (1) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
/// NB: this can be used only for smooth functions.
/// NB: if significantly bigger than zero, then the Hessian is not correctly computed.
///
NANO_PUBLIC scalar_t hess_accuracy(const function_t&, const vector_t& x, scalar_t early_stopping_epsilon = 1e-10);

///
/// \brief return the maximum violation (if any) of the convexity inequality along the [x1, x2] line.
///
/// NB: if non-zero, then the function is not convex.
/// NB: the strong-convexity coefficient is used as well.
///
NANO_PUBLIC scalar_t convex_accuracy(const function_t&, const vector_t& x1, const vector_t& x2, int steps);

///
/// \brief transform in-place the given (A, b) so that the equality constraint `Ax = b` is full row rank
///     (thus the rows are linearly independant).
///
struct full_rank_stats_t
{
    tensor_size_t m_rank{0};        ///<
    bool          m_changed{false}; ///<
};

NANO_PUBLIC full_rank_stats_t make_full_rank(matrix_t& A, vector_t& b);

///
/// \brief transform in-place the given (A, b) so that all rows have at least a non-zero component.
///
/// NB: this is useful for both equality `Ax = b` and inequality `Ax <= b` constraints.
/// NB: also verify if the right-hand component is consistent in this case
///     (e.g. `b_i` must be zero for equality constraints and positive for inequality constraints).
///
struct zero_rows_stats_t
{
    tensor_size_t m_removed{0};      ///< how many zero rows have been removed from (A, b)
    tensor_size_t m_inconsistent{0}; ///< how many removed rows were inconsistent
};

NANO_PUBLIC zero_rows_stats_t remove_zero_rows_equality(matrix_t& A, vector_t& b);
NANO_PUBLIC zero_rows_stats_t remove_zero_rows_inequality(matrix_t& A, vector_t& b);

///
/// \brief returns true whether the given quadratic term is convex (aka positive semi-definite).
///
NANO_PUBLIC bool is_convex(const matrix_t&, scalar_t tol = 1e-10);

///
/// \brief returns the strong convexity factor of a quadratic term, or zero if not convex.
///
NANO_PUBLIC scalar_t strong_convexity(const matrix_t&);

///
/// \brief return a compact linear representation (A, b, G, h) of the functional constraints (if possible):
///     Ax = b (gathers all equality constraints) and
///     Gx <= b (gathers all inequality constraints).
///
/// NB: if any constraint is not linear, then std::nullopt is returned.
///
struct linear_constraints_t
{
    matrix_t m_A; ///<
    vector_t m_b; ///<
    matrix_t m_G; ///<
    vector_t m_h; ///<
};

NANO_PUBLIC std::optional<linear_constraints_t> make_linear_constraints(const function_t&);
} // namespace nano
