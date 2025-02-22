#pragma once

#include <nano/function/linear.h>

namespace nano
{
///
/// \brief test/benchmark linear program from
///     exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a total budget constraint:
///  min  c.dot(x)
///  s.t. 1.dot(x) = alpha, 0 <= x <= 1
///  and  alpha is an integer between 0 and n.
///
/// NB: the implementation scales `alpha` to the number of dimensions `n`, thus it uses a dimension-free parameter in
/// the range [0, 1].
///
class NANO_PUBLIC linear_program_cvx48e_eq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48e_eq_t(tensor_size_t dims = 10, scalar_t alpha = 0.0);

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

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a total budget constraint:
///  min  c.dot(x)
///  s.t. 1.dot(x) <= alpha, 0 <= x <= 1
///  and  alpha is an integer between 1 and n.
///
/// NB: the implementation scales `alpha` to the number of dimensions `n`, resulting in (0, 1].
///
class NANO_PUBLIC linear_program_cvx48e_ineq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48e_ineq_t(tensor_size_t dims = 10, scalar_t alpha = 1e-6);

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
