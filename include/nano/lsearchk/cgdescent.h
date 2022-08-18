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
    /// NB: The implementation follows the notation from (1).
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

            const solver_state_t& state0; ///< original point
            solver_state_t&       c;      ///< tentative point
            lsearch_step_t        a, b;   ///< lower/upper bounds of the bracketing interval
        };

        bool done(const state_t&, scalar_t c1, scalar_t c2, scalar_t epsilonk, bool bracketed = true);

        void move(state_t&, scalar_t t) const;
        void update(state_t&, scalar_t epsilonk, scalar_t theta, int max_iterations) const;
        void updateU(state_t&, scalar_t epsilonk, scalar_t theta, int max_iterations) const;
        void bracket(state_t&, scalar_t ro, scalar_t epsilonk, scalar_t theta, int max_iterations) const;
    };
} // namespace nano
