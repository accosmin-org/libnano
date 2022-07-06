#pragma once

#include <nano/function.h>

namespace nano
{
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

    ///
    /// \brief constructs an equality constraint: h(x) = 0.
    ///
    NANO_PUBLIC constraint_t make_equality_constraint(rfunction_t&& constraint);

    ///
    /// \brief constructs an inequality constraint: g(x) <= 0.
    ///
    NANO_PUBLIC constraint_t make_inequality_constraint(rfunction_t&& constraint);

    ///
    /// \brief constructs an affine equality constraint: h(x) = q.dot(x) + r = 0.
    ///
    NANO_PUBLIC constraint_t make_affine_equality_constraint(vector_t q, scalar_t r);

    ///
    /// \brief constructs an affine inequality constraint: g(x) = q.dot(x) + r <= 0.
    ///
    NANO_PUBLIC constraint_t make_affine_inequality_constraint(vector_t q, scalar_t r);

    ///
    /// \brief constructs a quadratic equality constraint: h(x) = 1/2 * x.dot(P * x) + q.dot(x) + r = 0.
    ///
    NANO_PUBLIC constraint_t make_quadratic_equality_constraint(matrix_t P, vector_t q, scalar_t r);

    ///
    /// \brief constructs a quadratic inequality constraint: h(x) = 1/2 * x.dot(P * x) + q.dot(x) + r <= 0.
    ///
    NANO_PUBLIC constraint_t make_quadratic_inequality_constraint(matrix_t P, vector_t q, scalar_t r);

    ///
    /// \brief constructs a box constraint per dimension: min_i <= x_i <= max_i.
    ///
    NANO_PUBLIC constraints_t make_box_constraints(vector_t min, vector_t max);

    ///
    /// \brief registers a box constraint for all dimensions: min <= x_i <= max, for 0 <= i < size.
    ///
    NANO_PUBLIC constraints_t make_box_constraints(scalar_t min, scalar_t max, tensor_size_t size);

    ///
    /// \brief registers a ball constraint: g(x) = ||x - origin||^2 <= radius^2.
    ///
    NANO_PUBLIC constraint_t make_ball_constraint(vector_t origin, scalar_t radius);

    ///
    /// \brief models a hyper-ball (equality or inequality) constraint: c(x) = ||x - origin||^2 - radius^2.
    ///
    class NANO_PUBLIC ball_constraint_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        ball_constraint_t(vector_t origin, scalar_t radius);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        vector_t m_origin;      ///<
        scalar_t m_radius{0.0}; ///<
    };

    ///
    /// \brief models an affine (equality or inequality) constraint: c(x) = q.dot(x) + r.
    ///
    class NANO_PUBLIC affine_constraint_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        affine_constraint_t(vector_t q, scalar_t r);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        vector_t m_q;      ///<
        scalar_t m_r{0.0}; ///<
    };

    ///
    /// \brief models a quadratic (equality or inequality) constraint: c(x) = 1/2 * x.dot(P * x) + q.dot(x) + r.
    ///
    class NANO_PUBLIC quadratic_constraint_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        quadratic_constraint_t(matrix_t P, vector_t q, scalar_t r);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        matrix_t m_P;      ///<
        vector_t m_q;      ///<
        scalar_t m_r{0.0}; ///<
    };
} // namespace nano
