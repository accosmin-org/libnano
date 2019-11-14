#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief numerical optimization state described by:
    ///     current point (x),
    ///     function value (f),
    ///     gradient (g),
    ///     descent direction (d),
    ///     line-search step (t).
    ///
    class solver_state_t
    {
    public:

        enum class status
        {
            converged,      ///< convergence criterion reached
            max_iters,      ///< maximum number of iterations reached without convergence (default)
            failed,         ///< optimization failed (e.g. line-search failed)
            stopped         ///< user requested stop
        };

        ///
        /// \brief default constructor
        ///
        solver_state_t() = default;

        ///
        /// \brief constructor
        ///
        template <typename tvector>
        solver_state_t(const function_t& ffunction, tvector x0) :
            function(&ffunction),
            x(std::move(x0)),
            g(vector_t::Zero(x.size())),
            d(vector_t::Zero(x.size()))
        {
            f = function->vgrad(x, &g);
        }

        ///
        /// \brief move to another point
        ///
        template <typename tvector>
        bool update(const tvector& xx)
        {
            assert(function);
            assert(x.size() == xx.size());
            assert(x.size() == function->size());
            x = xx;
            f = function->vgrad(x, &g);
            return static_cast<bool>(*this);
        }

        ///
        /// \brief line-search step along the descent direction of state0
        ///
        bool update(const solver_state_t& state0, const scalar_t tt)
        {
            t = tt;
            return update(state0.x + t * state0.d);
        }

        ///
        /// \brief check convergence: the gradient is relatively small
        ///
        bool converged(const scalar_t epsilon) const
        {
            return convergence_criterion() < epsilon;
        }

        ///
        /// \brief convergence criterion: relative gradient
        ///
        scalar_t convergence_criterion() const
        {
            return g.lpNorm<Eigen::Infinity>() / std::max(scalar_t(1), std::fabs(f));
        }

        ///
        /// \brief check divergence
        ///
        operator bool() const // NOLINT(hicpp-explicit-conversions)
        {
            return std::isfinite(t) && std::isfinite(f) && std::isfinite(convergence_criterion());
        }

        ///
        /// \brief compute the dot product between the gradient and the descent direction
        ///
        auto dg() const
        {
            return g.dot(d);
        }

        ///
        /// \brief check if the chosen direction is a descent direction
        ///
        auto has_descent() const
        {
            return dg() < 0;
        }

        ///
        /// \brief check if the current step satisfies the Armijo condition (sufficient decrease)
        ///
        bool has_armijo(const solver_state_t& state0, const scalar_t c1) const
        {
            assert(c1 > 0 && c1 < 1);
            return f <= state0.f + t * c1 * state0.dg();
        }

        ///
        /// \brief check if the current step satisfies the approximate Armijo condition (sufficient decrease)
        ///     see CG_DESCENT
        ///
        bool has_approx_armijo(const solver_state_t& state0, const scalar_t epsilon) const
        {
            return f <= state0.f + epsilon;
        }

        ///
        /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature)
        ///
        bool has_wolfe(const solver_state_t& state0, const scalar_t c2) const
        {
            assert(c2 > 0 && c2 < 1);
            return dg() >= c2 * state0.dg();
        }

        ///
        /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature)
        ///
        bool has_strong_wolfe(const solver_state_t& state0, const scalar_t c2) const
        {
            assert(c2 > 0 && c2 < 1);
            return std::fabs(dg()) <= c2 * std::fabs(state0.dg());
        }

        ///
        /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature)
        ///     see CG_DESCENT
        ///
        bool has_approx_wolfe(const solver_state_t& state0, const scalar_t c1, const scalar_t c2) const
        {
            assert(0 < c1 && c1 < scalar_t(0.5) && c1 < c2 && c2 < 1);
            return (2 * c1 - 1) * state0.dg() >= dg() && dg() >= c2 * state0.dg();
        }

        // attributes
        const function_t*   function{nullptr};      ///<
        vector_t            x, g, d;                ///< parameter, gradient, descent direction
        scalar_t            f{0}, t{0};             ///< function value, step size
        status              m_status{status::max_iters};    ///< optimization status
        size_t              m_fcalls{0};            ///< #function value evaluations so far
        size_t              m_gcalls{0};            ///< #function gradient evaluations so far
        size_t              m_iterations{0};        ///< #optimization iterations so far
    };

    template <>
    inline enum_map_t<solver_state_t::status> enum_string<solver_state_t::status>()
    {
        return
        {
            { solver_state_t::status::converged,   "converged" },
            { solver_state_t::status::max_iters,   "max_iters" },
            { solver_state_t::status::failed,      "failed" },
            { solver_state_t::status::stopped,     "stopped" }
        };
    }

    inline bool operator<(const solver_state_t& one, const solver_state_t& two)
    {
        return  (std::isfinite(one.f) ? one.f : std::numeric_limits<scalar_t>::max()) <
                (std::isfinite(two.f) ? two.f : std::numeric_limits<scalar_t>::max());
    }

    inline std::ostream& operator<<(std::ostream& os, const solver_state_t::status status)
    {
        return os << scat(status);
    }
}
