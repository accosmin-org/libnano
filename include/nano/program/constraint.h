#pragma once

#include <nano/eigen.h>
#include <nano/tensor/stack.h>

namespace nano::program
{
///
/// \brief models a linear constraint: A * x ? b.
///
struct NANO_PUBLIC constraint_t
{
    ///
    /// \brief constructor
    ///
    constraint_t();
    constraint_t(matrix_t A, vector_t b);
    constraint_t(const vector_t& a, scalar_t b);

    ///
    /// \brief return true if the constraint is given.
    ///
    explicit operator bool() const;

    // attributes
    matrix_t m_A; ///<
    vector_t m_b; ///<
};

///
/// \brief models a linear equality constraint: A * x = b.
///
struct NANO_PUBLIC equality_t : public constraint_t
{
    using constraint_t::constraint_t;

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(const vector_t& x, scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon()) const;

    ///
    /// \brief return the deviation of the given point from the equality constraint.
    ///
    template <typename tvector, std::enable_if_t<is_eigen_v<tvector>, bool> = true>
    scalar_t deviation(const tvector& x) const
    {
        return this->operator bool() ? deviation(m_A, m_b, x) : std::numeric_limits<scalar_t>::quiet_NaN();
    }

    template <typename tvector, std::enable_if_t<is_eigen_v<tvector>, bool> = true>
    static scalar_t deviation(const matrix_t& A, const vector_t& b, const tvector& x)
    {
        return (A * x - b).array().abs().maxCoeff();
    }
};

///
/// \brief models a linear inequality constraint: A * x <= b (element-wise).
///
struct NANO_PUBLIC inequality_t : public constraint_t
{
    using constraint_t::constraint_t;

    ///
    /// \brief contruct from one-sided contraint: lower <= x (element-wise).
    ///
    template <typename tlower, std::enable_if_t<is_eigen_v<tlower>, bool> = true>
    static inequality_t greater(const tlower& lower)
    {
        const auto dims = lower.size();

        return {-matrix_t::Identity(dims, dims), -lower};
    }

    static inequality_t greater(const tensor_size_t dims, const scalar_t lower)
    {
        return {-matrix_t::Identity(dims, dims), -vector_t::Constant(dims, lower)};
    }

    ///
    /// \brief construct from rectangle contraints: lower <= x <= upper (element-wise).
    ///
    template <typename tlower, typename tupper, std::enable_if_t<is_eigen_v<tlower>, bool> = true,
              std::enable_if_t<is_eigen_v<tupper>, bool> = true>
    static inequality_t from_rectangle(const tlower& lower, const tupper& upper)
    {
        assert(lower.size() == upper.size());

        const auto dims = lower.size();

        return {stack<scalar_t>(2 * dims, dims, matrix_t::Identity(dims, dims), -matrix_t::Identity(dims, dims)),
                stack<scalar_t>(2 * dims, upper, -lower)};
    }

    static inequality_t from_rectangle(const tensor_size_t dims, const scalar_t lower, const scalar_t upper)
    {
        return from_rectangle(vector_t::Constant(dims, lower), vector_t::Constant(dims, upper));
    }

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(const vector_t& x, scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon()) const;

    ///
    /// \brief return the deviation of the given point from the inequality constraint.
    ///
    template <typename tvector, std::enable_if_t<is_eigen_v<tvector>, bool> = true>
    scalar_t deviation(const tvector& x) const
    {
        return this->operator bool() ? (m_A * x - m_b).array().maxCoeff() : std::numeric_limits<scalar_t>::quiet_NaN();
    }

    ///
    /// \brief return a strictly feasible point, if possible.
    ///
    std::optional<vector_t> make_strictly_feasible() const;
};

///
/// \brief models a linearly-constrained programming problem:
///     min  f(x)
///     s.t. A * x = b and G * x <= h.
///
struct NANO_PUBLIC linear_constrained_t
{
    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(const vector_t& x, scalar_t epsilon = std::numeric_limits<scalar_t>::epsilon()) const;

    ///
    /// \brief return true if the equality constraint `Ax = b` is not full row rank.
    ///
    /// in this case the constraint is transformed in-place to obtain row-independant linear constraints
    ///     by performing an appropriate matrix decomposition.
    ///
    bool reduce();

    // attributes
    equality_t   m_eq;   ///<
    inequality_t m_ineq; ///<
};

NANO_PUBLIC equality_t   operator&(const equality_t&, const equality_t&);
NANO_PUBLIC inequality_t operator&(const inequality_t&, const inequality_t&);
} // namespace nano::program
