#pragma once

#include <nano/function/quadratic.h>

namespace nano
{
///
/// \brief test/benchmark quadratic program from
///     appendix A.2, "OSQP: an operator splitting solver for quadratic programs", B. Stellato et al, 2020
///
/// minimize a convex quadratic function:
///     min. 1/2 * x.dot(P * x) + q.dot(x)
///     s.t. A * x = b.
///
class NANO_PUBLIC quadratic_program_eqcqp_t final : public quadratic_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit quadratic_program_eqcqp_t(tensor_size_t dims = 10, scalar_t neqs = 0.5, scalar_t alpha = 1e-2);

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
