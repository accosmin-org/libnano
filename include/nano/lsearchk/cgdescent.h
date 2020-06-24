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
            wolfe,                  ///< criterion v1: Wolfe conditions
            approx_wolfe,           ///< criterion V2: approximate Wolfe conditions
            wolfe_approx_wolfe      ///< criterion V3: Wolfe conditions until the function changes slowly
                                    ///<    then switch to approximate Wolfe conditions
        };

        ///
        /// \brief default constructor
        ///
        lsearchk_cgdescent_t() = default;

        ///
        /// \brief @see lsearchk_t
        ///
        [[nodiscard]] rlsearchk_t clone() const final;

        ///
        /// \brief @see lsearchk_t
        ///
        bool get(const solver_state_t& state0, solver_state_t& state) final;

        ///
        /// \brief change parameters
        ///
        void epsilon(const scalar_t epsilon) { m_epsilon = epsilon; }
        void theta(const scalar_t theta) { m_theta = theta; }
        void gamma(const scalar_t gamma) { m_gamma = gamma; }
        void delta(const scalar_t delta) { m_delta = delta; }
        void omega(const scalar_t omega) { m_omega = omega; }
        void ro(const scalar_t ro) { m_ro = ro; }
        void crit(const criterion crit) { m_criterion = crit; }

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto epsilon() const { return m_epsilon.get(); }
        [[nodiscard]] auto theta() const { return m_theta.get(); }
        [[nodiscard]] auto gamma() const { return m_gamma.get(); }
        [[nodiscard]] auto delta() const { return m_delta.get(); }
        [[nodiscard]] auto omega() const { return m_omega.get(); }
        [[nodiscard]] auto ro() const { return m_ro.get(); }
        [[nodiscard]] auto crit() const { return m_criterion; }

    private:

        enum class status
        {
            exit,           ///< exit criterion generated (Wolfe + approximate Wolfe)
            fail,           ///< search failed
            done            ///< search succeeded, apply next step
        };

        bool evaluate(const solver_state_t&, const solver_state_t&);
        bool evaluate(const solver_state_t&, scalar_t, solver_state_t&);

        status update(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        status updateU(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        status secant2(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        status bracket(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);

        // attributes
        sparam1_t   m_epsilon{"lsearchk::cgdescent::epsilon", 0, LT, 1e-6, LT, 1e+6};   ///< see (2)
        sparam1_t   m_theta{"lsearchk::cgdescent::theta", 0, LT, 0.5, LT, 1};           ///< see (2)
        sparam1_t   m_gamma{"lsearchk::cgdescent::gamma", 0, LT, 0.66, LT, 1};          ///< see (2)
        sparam1_t   m_delta{"lsearchk::cgdescent::delta", 0, LT, 0.7, LT, 1};           ///< see (2)
        sparam1_t   m_omega{"lsearchk::cgdescent::omega", 0, LT, 1e-3, LT, 1};          ///< see (2)
        sparam1_t   m_ro{"lsearchk::cgdescent::ro", 1, LT, 5.0, LT, 1e+6};              ///< see (2)
        scalar_t    m_sumQ{0};                                                          ///< see (2)
        scalar_t    m_sumC{0};                                                          ///< see (2)
        bool        m_approx{false};                                                    ///< see (2)
        criterion   m_criterion{criterion::wolfe_approx_wolfe};                         ///<
    };

    template <>
    inline enum_map_t<lsearchk_cgdescent_t::criterion> enum_string<lsearchk_cgdescent_t::criterion>()
    {
        return
        {
            { lsearchk_cgdescent_t::criterion::wolfe,               "wolfe"},
            { lsearchk_cgdescent_t::criterion::approx_wolfe,        "approx_wolfe"},
            { lsearchk_cgdescent_t::criterion::wolfe_approx_wolfe,  "wolfe_approx_wolfe"}
        };
    }
}
