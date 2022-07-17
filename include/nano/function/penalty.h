#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief generic penalty function: q(c, x) = f(x) + c * sum(p(g_i(x)), i), where:
    ///     * f(x) is the objective function to minimize,
    ///     * c > 0 is the penalty term - the higher the better the penalty function approximates
    ///         the original constrained optimization problem,
    ///     * p(y) is the penalty function with p(y) = 0 whenever y <= 0 and p(y) > 0 otherwise,
    ///     * {h_j(x) == 0} is the set of equality constraints (each treated as two inequalities) and
    ///     * {g_i(x) <= 0} is the set of inequality constraints.
    ///
    /// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
    ///
    /// NB: each equality h(x) == 0 is treated as two inequalities: h(x) <= 0 and h(x) >= 0.
    /// NB: the penalty function is set to +infinity when at least one of the penalties is greater than the cutoff.
    /// NB: the cutoff constant forces the penalty function to be bounded whenever the objective function is bounded.
    ///
    class NANO_PUBLIC penalty_function_t : public function_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        explicit penalty_function_t(const function_t&);

        ///
        /// \brief set the cutoff constant.
        ///
        penalty_function_t& cutoff(scalar_t);

        ///
        /// \brief set the penalty term.
        ///
        penalty_function_t& penalty(scalar_t);

        ///
        /// \brief returns the cutoff constant.
        ///
        auto cutoff() const { return m_cutoff; }

        ///
        /// \brief returns the penalty term.
        ///
        auto penalty() const { return m_penalty; }

        ///
        /// \brief returns the constrained optimization objective.
        ///
        const auto& function() const { return m_function; }

        ///
        /// \brief returns the set of registered constraints.
        ///
        const constraints_t& constraints() const override { return m_function.constraints(); }

    private:
        // attributes
        const function_t& m_function;     ///<
        scalar_t          m_penalty{1.0}; ///<
        scalar_t          m_cutoff{1e+3}; ///<
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
    /// see "On smoothing exact penalty functions for convex constrained optimization", by M. Pinar, S. Zenios, 1994
    ///
    class NANO_PUBLIC linear_quadratic_penalty_function_t final : public penalty_function_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        explicit linear_quadratic_penalty_function_t(const function_t&);

        ///
        /// \brief set the smoothing factor.
        ///
        linear_quadratic_penalty_function_t& smoothing(scalar_t);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx = nullptr) const override;

    private:
        // attributes
        scalar_t m_smoothing{1.0}; ///<
    };
} // namespace nano
