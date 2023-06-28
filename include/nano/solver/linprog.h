#pragma once

#include <nano/arch.h>
#include <nano/eigen.h>

namespace nano
{
///
/// \brief the standard form of linear programming:
///     f(x) = c.dot(x) s.t Ax = b and x >= 0,
///
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
struct NANO_PUBLIC linear_program_t
{
    ///
    /// \brief constructor
    ///
    linear_program_t(vector_t c, matrix_t A, vector_t b);

    ///
    /// \brief return true if the given
    ///
    bool feasible(const vector_t& x, scalar_t epsilon) const;

    // attributes
    vector_t m_c; ///<
    matrix_t m_A; ///<
    vector_t m_b; ///<
};

///
/// \brief return a starting point appropriate for primal-dual interior point methods.
///
/// see ch.14 (page 410) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
NANO_PUBLIC std::tuple<vector_t, vector_t, vector_t> make_starting_point(const linear_program_t&);
} // namespace nano
