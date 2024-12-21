#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief models the general quadratic programs:
///     min  f(x) = 1/2 * x.dot(Q * x) + c.dot(x)
///     s.t. A * x = b
///     and  G * x <= h.
///
/// NB: the equality and the inequality constraints are optional.
///
/// see (1) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
/// see (2) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
class NANO_PUBLIC quadratic_program_t : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    quadratic_program_t(string_t id, matrix_t Q, vector_t c);

    ///
    /// \brief constructor (use the upper triangular representation of a symmetric Q)
    ///
    quadratic_program_t(string_t id, const vector_t& Q_upper_triangular, vector_t c);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

    ///
    /// \brief change the objective with a compatible one and keep the constraints.
    ///
    void reset(matrix_t Q, vector_t c);
    void reset(matrix_t Q);

private:
    // attributes
    matrix_t m_Q; ///<
    vector_t m_c; ///<
};
} // namespace nano
