#pragma once

#include <nano/function/quadratic.h>

namespace nano
{
///
/// \brief test/benchmark quadratic program from:
///     appendix A.2, "OSQP: an operator splitting solver for quadratic programs", B. Stellato et al, 2020
///
/// minimize a convex quadratic function:
///     min. 1/2 * x.dot(P * x) + q.dot(x)
///     s.t. A * x = b.
///
/// NB: b is generated so that the equality constrained can be solved exactly.
///
class NANO_PUBLIC equality_constrained_quadratic_program_t final : public quadratic_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit equality_constrained_quadratic_program_t(tensor_size_t dims = 10, const function_t* reference = nullptr);

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

private:
    void generate(const function_t& reference);
};
} // namespace nano
