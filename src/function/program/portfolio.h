#pragma once

#include <nano/function/quadratic.h>

namespace nano
{
///
/// \brief test/benchmark quadratic program from
///     appendix A.4, "OSQP: an operator splitting solver for quadratic programs", B. Stellato et al, 2020
///
/// minimize a convex quadratic function (portfolio optimization):
///     min. gamma * x.dot(SIGMA * x) - miu.dot(x)
///     s.t. 1.dot(x) = 1,
///          x >= 0.
///
class NANO_PUBLIC quadratic_program_portfolio_t final : public quadratic_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit quadratic_program_portfolio_t(tensor_size_t dims = 10, scalar_t factors = 0.5, scalar_t gamma = 1.0);

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
