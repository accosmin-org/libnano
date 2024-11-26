#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief models the class of linear programs.
///
/// general form (equality constraint, inequality constraint):
///     min c.dot(x)
///     s.t A * x = b
///     and G * x <= h.
///
/// standard form (equality constraint, no inequality constraint):
///     min c.dot(x)
///     s.t A * x = b
///     and x >= 0.0.
///
/// inequality form (no equality constraint, inequality constraint):
///     min c.dot(x)
///     s.t A * x <= b.
///
/// rectangle-inequality form (no equality constraint, inequality constraint):
///     min c.dot(x)
///     s.t l <= x <= u.
///
/// NB: the equality and the inequality constraints are optional.
///
/// see (1) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
/// see (2) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
class NANO_PUBLIC linear_program_t final : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_t(vector_t c);

private:
    // attributes
    vector_t m_c; ///<
    matrix_t m_A; ///<
    vector_t m_b; ///<
    matrix_t m_G; ///<
    vector_t m_h; ///<
};

///
/// \brief construct a linear program from the given objective and the equality and inequality constraints.
///
template <class... tconstraints>
auto make_linear(vector_t c, const tconstraints&... constraints)
{
    auto program = linear_program_t{std::move(c)};
    program.constrain(constraints...);
    return program;
}
} // namespace nano::program
