#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief generic penalty function: q(c, x) = f(x) + c * sum(h_j(x)^2, j) + c * sum(p(g_i(x)), i), where:
    ///     * f(x) is the objective function to minimize,
    ///     * c > 0 is the penalty term - the higher the better the penalty function approximates
    ///         the original constrained optimization problem,
    ///     * p(y) is the penalty function with p(y) = 0 whenever y <= 0 and p(y) > 0 otherwise,
    ///     * {h_j(x) == 0} is the set of equality constraints (always squared) and
    ///     * {g_i(x) <= 0} is the set of inequality constraints.
    ///
    ///     see "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
    ///
    class NANO_PUBLIC penalty_function_t : public function_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        explicit penalty_function_t(const function_t&);

        ///
        /// \brief set the penalty term.
        ///
        penalty_function_t& penalty_term(scalar_t);

        ///
        /// \brief returns the penalty term.
        ///
        auto penalty_term() const { return m_penalty_term; }

        ///
        /// \brief returns the constrained optimization objective.
        ///
        const auto& function() const { return m_function; }

    private:
        // attributes
        const function_t& m_function;              ///<
        scalar_t          m_penalty_term{1.0};     ///<
    };

    ///
    /// \brief linear penalty function: p(y) = max(0, y).
    ///
    class NANO_PUBLIC linear_penalty_function_t final : public penalty_function_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        explicit linear_penalty_function_t(const function_t&);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;
    };

    ///
    /// \brief quadratic penalty function: p(y) = max(0, y)^2.
    ///
    class NANO_PUBLIC quadratic_penalty_function_t final : public penalty_function_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        explicit quadratic_penalty_function_t(const function_t&);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;
    };

    ///
    /// \brief epsilon-smoothed linear quadratic penalty function: p(y, epsilon) =
    ///     * 0, if y <= 0,
    ///     * y^2 / (2 * epsilon), if 0 <= y <= epsilon,
    ///     * y - epsilon / 2, if y >= epsilon.
    ///
    ///     see "On smoothing exact penalty functions for convex constrained optimization", by M. Pinar, S. Zenios, 1994
    ///
    class NANO_PUBLIC linear_quadratic_penalty_function_t final : public penalty_function_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        explicit linear_quadratic_penalty_function_t(const function_t&);

        ///
        /// \brief set the smoothing factor (if applicable).
        ///
        linear_quadratic_penalty_function_t& smoothing_factor(scalar_t);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        scalar_t m_smoothing_factor{1.0}; ///<
    };
} // namespace nano
