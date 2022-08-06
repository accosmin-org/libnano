#pragma once

#include <nano/lsearchk.h>

namespace nano
{
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
        /// \brief convergence criterion following (2).
        ///
        enum class criterion
        {
            wolfe,             ///< criterion v1: Wolfe conditions
            approx_wolfe,      ///< criterion V2: approximate Wolfe conditions
            wolfe_approx_wolfe ///< criterion V3: Wolfe conditions until the function changes slowly
                               ///<    then switch to approximate Wolfe conditions
        };

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
        enum class status
        {
            exit, ///< exit criterion generated (Wolfe + approximate Wolfe)
            fail, ///< search failed
            done  ///< search succeeded, apply next step
        };

        bool evaluate(const solver_state_t&, const solver_state_t&);
        bool evaluate(const solver_state_t&, scalar_t, solver_state_t&);

        status update(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        status updateU(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        status secant2(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        status bracket(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);

        // attributes
        scalar_t m_sumQ{0};       ///< see (2)
        scalar_t m_sumC{0};       ///< see (2)
        bool     m_approx{false}; ///< see (2)
    };

    template <>
    inline enum_map_t<lsearchk_cgdescent_t::criterion> enum_string<lsearchk_cgdescent_t::criterion>()
    {
        return {
            {             lsearchk_cgdescent_t::criterion::wolfe,              "wolfe"},
            {      lsearchk_cgdescent_t::criterion::approx_wolfe,       "approx_wolfe"},
            {lsearchk_cgdescent_t::criterion::wolfe_approx_wolfe, "wolfe_approx_wolfe"}
        };
    }
} // namespace nano
