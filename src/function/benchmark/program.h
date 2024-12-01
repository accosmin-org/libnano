#pragma once

#include <nano/program/expected.h>
#include <nano/program/linear.h>
#include <nano/program/quadratic.h>

namespace nano::program
{
using expected_linear_program_t    = std::tuple<linear_program_t, expected_t>;
using expected_quadratic_program_t = std::tuple<quadratic_program_t, expected_t>;

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (b), see "Convex Optimization", by S. Boyd and L. Vanderberghe.
///
/// minimize a linear function over a halfspace:
///     min  c.dot(x)
///     s.t. a.dot(x) <= b,
///     s.t. c = lambda * a
///     and  lambda <= 0.0.
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx48b(tensor_size_t dims, scalar_t lambda);

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (c), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a rectangle:
///     min  c.dot(x)
///     s.t. l <= x <= u.
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx48c(tensor_size_t dims);

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over the probability simplex:
///     min  c.dot(x)
///     s.t. 1.dot(x) = 1, x >= 0.
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx48d_eq(tensor_size_t dims);

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over the probability simplex:
///     min  c.dot(x)
///     s.t. 1.dot(x) <= 1, x >= 0.
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx48d_ineq(tensor_size_t dims);

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a total budget constraint:
///  min  c.dot(x)
///  s.t. 1.dot(x) = alpha, 0 <= x <= 1
///  and  alpha is an integer between 0 and n.
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx48e_eq(tensor_size_t dims, tensor_size_t alpha);

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a total budget constraint:
///  min  c.dot(x)
///  s.t. 1.dot(x) <= alpha, 0 <= x <= 1
///  and  alpha is an integer between 0 and n.
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx48e_ineq(tensor_size_t dims, tensor_size_t alpha);

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (f), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a weighted budget constraint:
///  min  c.dot(x)
///  s.t. d.dot(x) = alpha * 1.dot(d), 0 <= x <= 1
///  and  d > 0 and 0 <= alpha <= 1.
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx48f(tensor_size_t dims, scalar_t alpha);

///
/// \brief test/benchmark linear program from
///     exercise 4.9, see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize the square linear program:
///  min  c.dot(x)
///  s.t. Ax <= b
///  and  A is square and nonsingular and A^T * c <= 0 (to be feasible).
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx49(tensor_size_t dims);

///
/// \brief test/benchmark linear program from
///     exercise 4.10, see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// standard form linear program:
///  min  c.dot(x)
///  s.t. Ax = b, x >= 0
///  and  the linear equality has exactly one solution (in this case A = D^t * D + I).
///
/// NB: b is generated as A * x, where x has all positive components (thus a feasible program)
///     if `feasible` is true, otherwise x has some negative components (thus an unfeasible program).
///
NANO_PUBLIC expected_linear_program_t make_linear_program_cvx410(tensor_size_t dims, bool feasible);

///
/// \brief test/benchmark quadratic program from
///     exercise 16.2, "Numerical optimization", Nocedal & Wright, 2nd edition
///
NANO_PUBLIC expected_quadratic_program_t make_quadratic_program_numopt162(tensor_size_t dims, tensor_size_t neqs);

///
/// \brief test/benchmark quadratic program from
///     exercise 16.25, "Numerical optimization", Nocedal & Wright, 2nd edition
///
NANO_PUBLIC expected_quadratic_program_t make_quadratic_program_numopt1625(tensor_size_t dims);
} // namespace nano::program
