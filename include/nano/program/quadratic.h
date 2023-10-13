#pragma once

#include <nano/program/constraint.h>

namespace nano::program
{
///
/// \brief models the general quadratic programs:
///     min f(x) = 1/2 * x.dot(Q * x) + c.dot(x)
///     s.t A * x = b
///     and G * x <= h.
///
/// NB: the equality and the inequality constraints are optional.
///
/// see (1) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
/// see (2) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
struct NANO_PUBLIC quadratic_program_t : public linear_constrained_t
{
    ///
    /// \brief constructor
    ///
    quadratic_program_t(matrix_t Q, vector_t c);

    ///
    /// \brief constructor (use the upper triangular representation of a symmetric Q)
    ///
    quadratic_program_t(const vector_t& Q_upper_triangular, vector_t c);

    ///
    /// \brief returns true if the quadratic program is convex (symmetric and positive semidefinite)
    ///
    bool convex() const;

    // attributes
    matrix_t m_Q; ///<
    vector_t m_c; ///<
};

NANO_PUBLIC quadratic_program_t operator&(const quadratic_program_t&, const equality_t&);
NANO_PUBLIC quadratic_program_t operator&(const quadratic_program_t&, const inequality_t&);
} // namespace nano::program
