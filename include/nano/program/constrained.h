#pragma once

#include <nano/program/stack.h>

namespace nano::program
{
template <class T>
inline constexpr bool is_constraint_v = is_equality<T>::value || is_inequality<T>::value;

///
/// \brief models a linearly-constrained programming problem:
///     min  f(x)
///     s.t. A * x = b and G * x <= h.
///
struct NANO_PUBLIC linear_constrained_t
{
    ///
    /// \brief in-place update both equality and inequality constraints (if given).
    ///
    /// NB: memory allocations are minimized when using appropriated Eigen operators to specify constraints
    ///     with utilities like `make_less`, `make_equality` or `make_greater`.
    ///
    template <class... tconstraints>
    void constrain(const tconstraints&... constraints)
    {
        static_assert((is_constraint_v<tconstraints> && ...));
        stack(m_eq.m_A, m_eq.m_b, m_ineq.m_A, m_ineq.m_b, constraints...);
    }

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(vector_cmap_t x, scalar_t epsilon = epsilon0<scalar_t>()) const;

    ///
    /// \brief return a strictly feasible point wrt inequality constraints, if possible.
    ///
    std::optional<vector_t> make_strictly_feasible() const;

    // attributes
    equality_t<matrix_t, vector_t>   m_eq;   ///<
    inequality_t<matrix_t, vector_t> m_ineq; ///<
};
} // namespace nano::program
