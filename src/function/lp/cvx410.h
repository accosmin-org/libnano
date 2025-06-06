#pragma once

#include <nano/function/linear.h>

namespace nano
{
///
/// \brief test/benchmark linear program from
///     exercise 4.10, see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// standard form linear program:
///     min. c.dot(x)
///     s.t. Ax = b, x >= 0
///     and  the linear equality has exactly one solution (in this case A = D^t * D + I).
///
/// NB: the vector `b` so that the equality constraints can be solved exactly.
///
class NANO_PUBLIC linear_program_cvx410_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx410_t(tensor_size_t dims = 10, uint64_t seed = 42);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    string_t do_name() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims) const override;
};
} // namespace nano
