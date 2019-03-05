#pragma once

#include <nano/solver.h>

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
    class lsearch_cgdescent_t final : public lsearch_strategy_t
    {
    public:

        lsearch_cgdescent_t() = default;
        void to_json(json_t&) const final;
        void from_json(const json_t&) final;
        bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

    private:

        bool evaluate(const solver_state_t&, const scalar_t,
            solver_state_t&);
        bool evaluate(const solver_state_t&, const scalar_t, const lsearch_step_t&, const lsearch_step_t&,
            solver_state_t&);

        bool update(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        bool updateU(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        bool secant2(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);
        bool bracket(const solver_state_t&, lsearch_step_t& a, lsearch_step_t& b, solver_state_t& c);

        static auto interpolate(const lsearch_step_t& a, const lsearch_step_t& b)
        {
            const auto tc = lsearch_step_t::cubic(a, b);
            const auto ts = lsearch_step_t::secant(a, b);
            const auto tb = lsearch_step_t::bisect(a, b);

            const auto tmin = std::min(a.t, b.t);
            const auto tmax = std::max(a.t, b.t);

            if (std::isfinite(tc) && tmin < tc && tc < tmax)
            {
                return tc;
            }
            else if (std::isfinite(ts) && tmin < ts && ts < tmax)
            {
                return ts;
            }
            else
            {
                return tb;
            }
        }

        // attributes
        scalar_t    m_epsilon0{static_cast<scalar_t>(1e-6)};///<
        scalar_t    m_epsilon{0};                           ///<
        scalar_t    m_theta{static_cast<scalar_t>(0.5)};    ///<
        scalar_t    m_gamma{static_cast<scalar_t>(0.66)};   ///<
        scalar_t    m_delta{static_cast<scalar_t>(0.7)};    ///<
        scalar_t    m_omega{static_cast<scalar_t>(1e-3)};   ///<
        scalar_t    m_ro{static_cast<scalar_t>(5.0)};       ///<
        scalar_t    m_sumQ{0};                              ///<
        scalar_t    m_sumC{0};                              ///<
        bool        m_approx{false};                        ///<
    };
}
