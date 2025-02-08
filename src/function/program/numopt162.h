#pragma once

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
///     and  1 <= neqs=A.rows() <= n.
///
/// NB: the implementation scales the number of equalities `neqs` to the number of dimensions `n`, thus it uses a
/// dimension-free parameter in the range (0, 1].
///
class NANO_PUBLIC quadratic_program_numopt162_t final : public quadratic_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit quadratic_program_numopt162_t(tensor_size_t dims = 10, scalar_t neqs = 1.0);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims) const override;
};
} // namespace nano
