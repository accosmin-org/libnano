#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief the standard form of linear programming:
///     f(x) = c.dot(x) s.t Ax = b and x >= 0,
///
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
class NANO_PUBLIC linprog_function_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    linprog_function_t(vector_t c, matrix_t A);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

private:
    // attributes
    vector_t m_c; ///<
    matrix_t m_A; ///<
};
} // namespace nano
