#pragma once

#include <nano/function/constraint.h>
#include <nano/string.h>

namespace nano
{
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
        function_t(const function_t&)            = delete;
        function_t& operator=(const function_t&) = delete;

        ///
        /// \brief enable moving
        ///
        function_t(function_t&&) noexcept;
        function_t& operator=(function_t&&) noexcept;

        ///
        /// \brief destructor
        ///
        virtual ~function_t();

        ///
        /// \brief function name to identify it in tests and benchmarks.
        ///
        string_t name(bool with_size = true) const;

        ///
        /// \brief returns the number of dimensions.
        ///
        tensor_size_t size() const { return m_size; }

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
        /// \brief register a constraint.
        ///
        /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
        ///
        bool constrain(constraint_t&&);
        bool constrain(scalar_t min, scalar_t max);
        bool constrain(scalar_t min, scalar_t max, tensor_size_t dimension);
        bool constrain(const vector_t& min, const vector_t& max);

        ///
        /// \brief returns the set of registered constraints.
        ///
        virtual const constraints_t& constraints() const;

        ///
        /// \brief evaluate the function's value at the given point
        ///     (and its gradient or sub-gradient if not smooth).
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const;

        ///
        /// \brief returns the number of function evaluation calls registered so far.
        ///
        tensor_size_t fcalls() const;

        ///
        /// \brief returns the number of function gradient calls registered so far.
        ///
        tensor_size_t gcalls() const;

        ///
        /// \brief clear collected statistics (e.g. function calls).
        ///
        void clear_statistics() const;

    protected:
        void convex(bool);
        void smooth(bool);
        void strong_convexity(scalar_t);

        virtual scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const = 0;

    private:
        // attributes
        string_t      m_name;              ///<
        tensor_size_t m_size{0};           ///< #free dimensions to optimize for
        bool          m_convex{false};     ///< whether the function is convex
        bool          m_smooth{false};     ///< whether the function is smooth (otherwise subgradients should be used)
        scalar_t      m_sconvexity{0};     ///< strong-convexity coefficient
        constraints_t m_constraints;       ///< optional equality and inequality constraints
        mutable tensor_size_t m_fcalls{0}; ///< #function value evaluations
        mutable tensor_size_t m_gcalls{0}; ///< #function gradient evaluations
    };
} // namespace nano
