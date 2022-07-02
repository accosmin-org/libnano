#pragma once

#include <memory>
#include <nano/arch.h>
#include <nano/eigen.h>
#include <nano/string.h>
#include <variant>

namespace nano
{
    class function_t;
    using rfunction_t  = std::unique_ptr<function_t>;
    using rfunctions_t = std::vector<rfunction_t>;

    struct minimum_t
    {
        scalar_t      m_value{0.0};
        tensor_size_t m_dimension{0};
    };

    struct maximum_t
    {
        scalar_t      m_value{0.0};
        tensor_size_t m_dimension{0};
    };

    struct equality_t
    {
        rfunction_t m_function;
    };

    struct inequality_t
    {
        rfunction_t m_function;
    };

    using constraint_t  = std::variant<minimum_t, maximum_t, equality_t, inequality_t>;
    using constraints_t = std::vector<constraint_t>;

    ///
    /// \brief generic multi-dimensional function typically used as the objective of a numerical optimization problem.
    ///
    /// optionally a set of equality and inequality constraints can be added
    /// following the generic constrained optimization problems:
    ///
    ///     argmin      f(x),           - the objective function
    ///     such that   h_j(x) = 0,     - the equality constraints
    ///                 g_i(x) <= 0.    - the inequality constraints
    ///
    /// NB: the (sub-)gradient of the function must be implemented.
    /// NB: the functions can be convex or non-convex and smooth or non-smooth.
    ///
    class NANO_PUBLIC function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        function_t(string_t name, tensor_size_t size);

        ///
        /// \brief disable copying
        ///
        function_t(const function_t&) = delete;
        function_t& operator=(const function_t&) = delete;

        ///
        /// \brief enable moving
        ///
        function_t(function_t&&) noexcept = default;
        function_t& operator=(function_t&&) noexcept = default;

        ///
        /// \brief destructor
        ///
        virtual ~function_t() = default;

        ///
        /// \brief function name to identify it in tests and benchmarks.
        ///
        string_t name(bool with_size = true) const;

        ///
        /// \brief returns the number of dimensions.
        ///
        tensor_size_t size() const { return m_size; }

        ///
        /// \brief compute the gradient accuracy (given vs. central finite difference approximation).
        ///
        scalar_t grad_accuracy(const vector_t& x) const;

        ///
        /// \brief check if the function is convex along the [x1, x2] line.
        ///
        bool is_convex(const vector_t& x1, const vector_t& x2, int steps) const;

        ///
        /// \brief returns whether the function is convex.
        ///
        bool convex() const { return m_convex; }

        ///
        /// \brief returns whether the function is smooth.
        ///
        /// NB: mathematically a smooth function is of C^inf class (gradients exist and are ontinuous for any order),
        /// but here it implies that the function is of C^1 class (differentiable and with continuous gradients)
        /// as required by the line-search methods. If this is not the case, either only sub-gradients are available
        /// of the gradients are not continuous.
        ///
        ///
        bool smooth() const { return m_smooth; }

        ///
        /// \brief returns the strong convexity coefficient.
        ///
        /// NB: if not convex, then the coefficient is zero.
        /// NB: it can be used to speed-up some numerical optimization algorithms if greater than zero.
        ///
        scalar_t strong_convexity() const { return m_sconvexity; }

        ///
        /// \brief returns true if the given point satisfies all the stored constraints.
        ///
        bool valid(const vector_t& x) const;

        ///
        /// \brief register a new equality constraint: h(x) = 0.
        ///
        /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
        ///
        bool constrain_equality(rfunction_t&& constraint);

        ///
        /// \brief register a new inequality constraint: g(x) <= 0.
        ///
        /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
        ///
        bool constrain_inequality(rfunction_t&& constraint);

        ///
        /// \brief registers a new linear equality constraint: h(x) = weights.dot(x) + bias = 0.
        ///
        /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
        ///
        bool constrain_equality(vector_t weights, scalar_t bias);

        ///
        /// \brief registers a new linear inequality constraint: g(x) = weights.dot(x) + bias <= 0.
        ///
        /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
        ///
        bool constrain_inequality(vector_t weights, scalar_t bias);

        ///
        /// \brief registers a new box constraint per dimension: min_i <= x_i <= max_i.
        ///
        /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
        ///
        bool constrain_box(vector_t min, vector_t max);

        ///
        /// \brief registers a new box constraint for all dimensions: min <= x_i <= max.
        ///
        /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
        ///
        bool constrain_box(scalar_t min, scalar_t max);

        ///
        /// \brief registers a new constraint: g(x) = ||x - origin||^2 <= radius^2.
        ///
        /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
        ///
        bool constrain_ball(vector_t origin, scalar_t radius);

        ///
        /// \brief returns the set of registered constraints.
        ///
        const constraints_t& constraints() const;

        ///
        /// \brief evaluate the function's value at the given point
        ///     (and its gradient or sub-gradient if not smooth).
        ///
        virtual scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const = 0;

    protected:
        void convex(bool);
        void smooth(bool);
        void strong_convexity(scalar_t);

    private:
        // attributes
        string_t      m_name;          ///<
        tensor_size_t m_size{0};       ///< #free dimensions to optimize for
        bool          m_convex{false}; ///< whether the function is convex
        bool          m_smooth{false}; ///< whether the function is smooth (otherwise subgradients should be used)
        scalar_t      m_sconvexity{0}; ///< strong-convexity coefficient
        constraints_t m_constraints;   ///< optional equality and inequality constraints
    };

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
    NANO_PUBLIC scalar_t valid(const vector_t&, const constraint_t&);

    ///
    /// \brief evaluate the given constraint's function value at the given point
    ///     (and its gradient or sub-gradient if not smooth).
    ///
    NANO_PUBLIC scalar_t vgrad(const constraint_t&, const vector_t& x, vector_t* gx = nullptr);
} // namespace nano
