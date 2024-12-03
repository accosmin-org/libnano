#pragma once

#include <nano/function/linear.h>
#include <nano/function/quadratic.h>

namespace nano
{
///
/// \brief test/benchmark quadratic program from
///     exercise 16.2, "Numerical optimization", Nocedal & Wright, 2nd edition
///
/// minimize a convex quadratic function:
///     min  1/2 * (x - x0).dot(x - x0)
///     s.t. A * x = b
///     and  1 <= neqs=A.rows() <= dims.
///
class NANO_PUBLIC quadratic_program_numopt162_t final : public quadratic_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit quadratic_program_numopt162_t(tensor_size_t dims = 10, tensor_size_t neqs = 10);

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
/// \brief test/benchmark quadratic program from
///     exercise 16.25, "Numerical optimization", Nocedal & Wright, 2nd edition
///
/// minimize a convex quadratic function:
///     min  1/2 * (x - xhat).dot(x - xhat)
///     s.t. l <= x <= u.
///
class NANO_PUBLIC quadratic_program_numopt1625_t final : public quadratic_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit quadratic_program_numopt1625_t(tensor_size_t dims = 10);

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
