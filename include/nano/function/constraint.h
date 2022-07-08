#pragma once

#include <memory>
#include <nano/arch.h>
#include <nano/eigen.h>
#include <variant>

namespace nano
{
    class function_t;
    using rfunction_t  = std::unique_ptr<function_t>;
    using rfunctions_t = std::vector<rfunction_t>;

    namespace constraint
    {
        struct ball_t
        {
            vector_t m_origin;
            scalar_t m_radius{0.0};
        };

        struct linear_t
        {
            vector_t m_q;
            scalar_t m_r{0.0};
        };

        struct quadratic_t
        {
            matrix_t m_P;
            vector_t m_q;
            scalar_t m_r{0.0};
        };

        struct functional_t
        {
            rfunction_t m_function;
        };

        ///
        /// \brief equality constraint: h(x) = ||x - origin||^2 - radius^2 <= 0
        ///
        struct ball_equality_t : ball_t
        {
        };

        ///
        /// \brief inequality constraint: g(x) = ||x - origin||^2 - radius^2 = 0
        ///
        struct ball_inequality_t : ball_t
        {
        };

        ///
        /// \brief equality constraint: h(x) = x(dimension) - constant = 0
        ///
        struct constant_t
        {
            scalar_t      m_value{0.0};
            tensor_size_t m_dimension{-1};
        };

        ///
        /// \brief inequality constraint: g(x) = constant - x(dimension) <= 0
        ///
        struct minimum_t : constant_t
        {
        };

        ///
        /// \brief inequality constraint: g(x) = x(dimension) - constant <= 0
        ///
        struct maximum_t : constant_t
        {
        };

        ///
        /// \brief equality constraint: h(x) = q.dot(x) + r = 0
        ///
        struct linear_equality_t : linear_t
        {
        };

        ///
        /// \brief inequality constraint: g(x) = q.dot(x) + r <= 0
        ///
        struct linear_inequality_t : linear_t
        {
        };

        ///
        /// \brief equality constraint: h(x) = 1/2 * x.dot(P * x) + q.dot(x) + r = 0
        ///
        struct quadratic_equality_t : quadratic_t
        {
        };

        ///
        /// \brief inequality constraint: q(x) = 1/2 * x.dot(P * x) + q.dot(x) + r = 0
        ///
        struct quadratic_inequality_t : quadratic_t
        {
        };

        ///
        /// \brief equality constraint: h(x) = 0
        ///
        struct functional_equality_t : functional_t
        {
        };

        ///
        /// \brief inequality constraint: g(x) <= 0
        ///
        struct functional_inequality_t : functional_t
        {
        };
    }; // namespace constraint

    ///
    /// \brief models a constraint that can be applied to an objective function.
    ///
    /// NB: the default constraint is by construction invalid.
    ///
    using constraint_t =
        std::variant<constraint::constant_t, constraint::minimum_t, constraint::maximum_t, constraint::ball_equality_t,
                     constraint::ball_inequality_t, constraint::linear_equality_t, constraint::linear_inequality_t,
                     constraint::quadratic_equality_t, constraint::quadratic_inequality_t,
                     constraint::functional_equality_t, constraint::functional_inequality_t>;

    using constraints_t = std::vector<constraint_t>;

    ///
    /// \brief returns whether the given constraint is convex.
    ///
    NANO_PUBLIC bool convex(const constraint_t&);

    ///
    /// \brief returns whether the given constraint is smooth.
    ///
    NANO_PUBLIC bool smooth(const constraint_t&);

    ///
    /// \brief returns whether the strong convexity coefficient of the given constraint.
    ///
    NANO_PUBLIC scalar_t strong_convexity(const constraint_t&);

    ///
    /// \brief returns how much a point violates the given constraint (the larger, the worse).
    ///
    NANO_PUBLIC scalar_t valid(const constraint_t&, const vector_t& x);

    ///
    /// \brief evaluate the given constraint's function value at the given point
    ///     (and its gradient or sub-gradient if not smooth).
    ///
    NANO_PUBLIC scalar_t vgrad(const constraint_t&, const vector_t& x, vector_t* gx = nullptr);

    ///
    /// \brief returns true if the given function and constrain are compatible.
    ///
    NANO_PUBLIC bool compatible(const constraint_t&, const function_t&);
} // namespace nano
