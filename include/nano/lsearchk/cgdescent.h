#pragma once

#include <nano/lsearchk.h>

namespace nano
{
    ///
    /// \brief convergence criterion for cgdescent line-search method following (2).
    ///
    enum class cgdescent_criterion_type
    {
        wolfe,             ///< criterion v1: Wolfe conditions
        approx_wolfe,      ///< criterion V2: approximate Wolfe conditions
        wolfe_approx_wolfe ///< criterion V3: Wolfe conditions until the function changes slowly
                           ///<    then switch to approximate Wolfe conditions
    };

    ///
    /// \brief CG_DESCENT:
    ///     see (1) "A new conjugate gradient method with guaranteed descent and an efficient line search",
    ///     by William W. Hager & HongChao Zhang, 2005
    ///
    ///     see (2) "Algorithm 851: CG_DESCENT, a Conjugate Gradient Method with Guaranteed Descent",
    ///     by William W. Hager & HongChao Zhang, 2006
    ///
    /// NB: The implementation follows the notation from (2).
    ///
    class NANO_PUBLIC lsearchk_cgdescent_t final : public lsearchk_t
    {
    public:
        ///
        /// \brief constructor
        ///
        lsearchk_cgdescent_t();

        ///
        /// \brief @see lsearchk_t
        ///
        rlsearchk_t clone() const final;

        ///
        /// \brief @see lsearchk_t
        ///
        bool get(const solver_state_t& state0, solver_state_t& state) final;

        ///
        /// \brief returns the constant used in the approximate Armijo condition.
        ///
        scalar_t approx_armijo_epsilon() const;

    private:
        struct state_t
        {
            state_t(const solver_state_t& state0, solver_state_t& state)
                : state0(state0)
                , c(state)
                , a(state0)
                , b(state)
            {
            }

            auto has_wolfe(scalar_t c1, scalar_t c2) const
            {
                return c.has_armijo(state0, c1) && c.has_wolfe(state0, c2);
            }

            auto has_approx_wolfe(scalar_t c1, scalar_t c2, scalar_t epsilonk) const
            {
                return c.has_approx_armijo(state0, epsilonk) && c.has_approx_wolfe(state0, c1, c2);
            }

            const solver_state_t& state0; ///< original point
            solver_state_t&       c;      ///< tentative point
            lsearch_step_t        a, b;   ///< lower/upper bounds of the bracketing interval
        };

        bool done(const state_t&, cgdescent_criterion_type, scalar_t c1, scalar_t c2, scalar_t epsilon, scalar_t omega,
                  bool bracketed = true);

        void move(state_t&, scalar_t t) const;
        void update(state_t&, scalar_t epsilon, scalar_t theta, int max_iterations) const;
        void updateU(state_t&, scalar_t epsilon, scalar_t theta, int max_iterations) const;
        void bracket(state_t&, scalar_t ro, scalar_t epsilon, scalar_t theta, int max_iterations) const;

        // attributes
        scalar_t m_Qk{0};         ///< see (2)
        scalar_t m_Ck{0};         ///< see (2)
        bool     m_approx{false}; ///< see (2)
    };

    template <>
    inline enum_map_t<cgdescent_criterion_type> enum_string<cgdescent_criterion_type>()
    {
        return {
            {             cgdescent_criterion_type::wolfe,              "wolfe"},
            {      cgdescent_criterion_type::approx_wolfe,       "approx_wolfe"},
            {cgdescent_criterion_type::wolfe_approx_wolfe, "wolfe_approx_wolfe"}
        };
    }
} // namespace nano
