#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief generic penalty function: q(c, x) = f(x) + c * sum(p(max(0, g_i(x))), i), where:
    ///
    ///     f(x) is the objective function to minimize,
    ///
    ///     c > 0 is the penalty term - the higher the better the penalty function approximates
    ///         the original constrained optimization problem,
    ///
    ///     p(y) is the penalty function with p(y) = 0 whenever y <= 0 and p(y) > 0 otherwise
    ///
    ///     and {g_i(x) <= 0} is the set of inequality constraints.
    ///
    /// NB: the equality constraints h(x) = 0 are transformed into two inequality constraints: h(x) <= 0 and -h(x) <= 0.
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
        const auto& constrained() const { return m_constrained; }

    private:
        // attributes
        const function_t& m_constrained;       ///<
        scalar_t          m_penalty_term{1.0}; ///<
    };

    ///
    /// \brief linear penalty function: q(c, x) = f(x) + c * sum(max(0, g_i(x)), i).
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
    /// \brief quadratic penalty function: q(c, x) = f(x) + c * sum(max(0, g_i(x)) ^ 2, i).
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
} // namespace nano
