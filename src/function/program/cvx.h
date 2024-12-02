#pragma once

#include <nano/function/linear.h>
#include <nano/function/quadratic.h>

namespace nano::program
{
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
class NANO_PUBLIC linear_program_cvx48b_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48b_t(tensor_size_t dims = 10, scalar_t lambda = -1.0);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (c), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a rectangle:
///     min  c.dot(x)
///     s.t. l <= x <= u.
///
class NANO_PUBLIC linear_program_cvx48c_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48c_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over the probability simplex:
///     min  c.dot(x)
///     s.t. 1.dot(x) = 1, x >= 0.
///
class NANO_PUBLIC linear_program_cvx48d_eq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48d_eq_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over the probability simplex:
///     min  c.dot(x)
///     s.t. 1.dot(x) <= 1, x >= 0.
///
class NANO_PUBLIC linear_program_cvx48d_ineq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48d_ineq_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a total budget constraint:
///  min  c.dot(x)
///  s.t. 1.dot(x) = alpha, 0 <= x <= 1
///  and  alpha is an integer between 0 and n.
///
class NANO_PUBLIC linear_program_cvx48e_eq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48e_eq_t(tensor_size_t dims = 10, tensor_size_t alpha = 0);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a total budget constraint:
///  min  c.dot(x)
///  s.t. 1.dot(x) <= alpha, 0 <= x <= 1
///  and  alpha is an integer between 0 and n.
///
class NANO_PUBLIC linear_program_cvx48e_ineq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48e_ineq_t(tensor_size_t dims = 10, tensor_size_t alpha = 0);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (f), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a weighted budget constraint:
///  min  c.dot(x)
///  s.t. d.dot(x) = alpha * 1.dot(d), 0 <= x <= 1
///  and  d > 0 and 0 <= alpha <= 1.
///
class NANO_PUBLIC linear_program_cvx48f_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48f_t(tensor_size_t dims = 10, scalar_t alpha = 0.5);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

///
/// \brief test/benchmark linear program from
///     exercise 4.9, see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize the square linear program:
///  min  c.dot(x)
///  s.t. Ax <= b
///  and  A is square and nonsingular and A^T * c <= 0 (to be feasible).
///
class NANO_PUBLIC linear_program_cvx49_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx49_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

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
class NANO_PUBLIC linear_program_cvx410_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx410_t(tensor_size_t dims = 10, bool feasible = true);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};
} // namespace nano::program
