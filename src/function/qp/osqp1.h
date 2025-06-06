#pragma once

#include <nano/function/quadratic.h>

namespace nano
{
///
/// \brief test/benchmark quadratic program from
///     appendix A.1, "OSQP: an operator splitting solver for quadratic programs", B. Stellato et al, 2020
///
/// minimize a convex quadratic function:
///     min. 1/2 * x.dot(P * x) + q.dot(x)
///     s.t. l <= A * x <= u.
///
/// NB: the implementation scales the number of inequalities `nineqs` to the number of dimensions `n`, thus it uses a
/// dimension-free parameter.
///
class NANO_PUBLIC quadratic_program_osqp1_t final : public quadratic_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit quadratic_program_osqp1_t(tensor_size_t dims = 10, uint64_t seed = 42, scalar_t nineqs = 10.0,
                                       scalar_t alpha = 1e-2);

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
