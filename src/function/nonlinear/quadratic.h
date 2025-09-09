#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief random quadratic function: f(x) = x.dot(a) + x * A * x, where A is PD.
///
class NANO_PUBLIC function_quadratic_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit function_quadratic_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_eval(eval_t) const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims) const override;

private:
    // attributes
    vector_t m_a;
    matrix_t m_A;
};
} // namespace nano
