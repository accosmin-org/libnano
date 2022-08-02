#pragma once

#include <nano/solver/state.h>

namespace nano
{
    ///
    /// \brief line-search step function:
    ///     phi(t) = f(x + t * d), f - the function to minimize and d - the descent direction.
    ///
    class lsearch_step_t
    {
    public:
        ///
        /// \brief interpolation method using the information at two trials.
        ///
        enum class interpolation
        {
            bisection, ///<
            quadratic, ///<
            cubic      ///<
        };

        ///
        /// \brief construction
        ///
        lsearch_step_t()                          = default;
        lsearch_step_t(lsearch_step_t&&) noexcept = default;
        lsearch_step_t(const lsearch_step_t&)     = default;

        // cppcheck-suppress noExplicitConstructor
        lsearch_step_t(const solver_state_t& state) // NOLINT(hicpp-explicit-conversions)
            : t(state.t)
            , f(state.f)
            , g(state.dg())
        {
        }

        lsearch_step_t(scalar_t tt, scalar_t ff, scalar_t gg)
            : t(tt)
            , f(ff)
            , g(gg)
        {
        }

        ///
        /// \brief assignment
        ///
        lsearch_step_t& operator=(lsearch_step_t&&) noexcept = default;
        lsearch_step_t& operator=(const lsearch_step_t&)     = default;

        lsearch_step_t& operator=(const solver_state_t& state)
        {
            t = state.t, f = state.f, g = state.dg();
            return *this;
        }

        ///
        /// \brief destructor
        ///
        ~lsearch_step_t() = default;

        ///
        /// \brief cubic interpolation of two line-search steps.
        ///     fit cubic: q(x) = a*x^3 + b*x^2 + c*x + d
        ///         given: q(u) = fu, q'(u) = gu
        ///         given: q(v) = fv, q'(v) = gv
        ///     minimizer: solution of 3*a*x^2 + 2*b*x + c = 0
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        static auto cubic(const lsearch_step_t& u, const lsearch_step_t& v)
        {
            const auto d1 = u.g + v.g - 3 * (u.f - v.f) / (u.t - v.t);
            const auto d2 = (v.t > u.t ? +1 : -1) * std::sqrt(d1 * d1 - u.g * v.g);
            return v.t - (v.t - u.t) * (v.g + d2 - d1) / (v.g - u.g + 2 * d2);
        }

        ///
        /// \brief quadratic interpolation of two line-search steps.
        ///     fit quadratic: q(x) = a*x^2 + b*x + c
        ///         given: q(u) = fu, q'(u) = gu
        ///         given: q(v) = fv
        ///     minimizer: -b/2a
        ///
        static auto quadratic(const lsearch_step_t& u, const lsearch_step_t& v, bool* convexity = nullptr)
        {
            const auto dt = u.t - v.t;
            const auto df = u.f - v.f;
            if (convexity != nullptr)
            {
                *convexity = (u.g - df / dt) * dt > 0;
            }
            return u.t - u.g * dt * dt / (2 * (u.g * dt - df));
        }

        ///
        /// \brief secant interpolation of two line-search steps.
        ///     fit quadratic: q(x) = a*x^2 + b*x + c
        ///         given: q'(u) = gu
        ///         given: q'(v) = gv
        ///     minimizer: -b/2a
        ///
        static auto secant(const lsearch_step_t& u, const lsearch_step_t& v)
        {
            return (v.t * u.g - u.t * v.g) / (u.g - v.g);
        }

        ///
        /// \brief bisection interpolation of two line-search steps.
        ///
        static auto bisection(const lsearch_step_t& u, const lsearch_step_t& v) { return (u.t + v.t) / 2; }

        ///
        /// \brief interpolation of two line-search steps.
        ///     first try a cubic interpolation, then a quadratic interpolation and finally do bisection
        ///         until the interpolated point is valid.
        ///
        static auto interpolate(const lsearch_step_t& u, const lsearch_step_t& v, interpolation method)
        {
            const auto tc = cubic(u, v);
            const auto tq = quadratic(u, v);
            const auto tb = bisection(u, v);

            switch (method)
            {
            case interpolation::cubic: return std::isfinite(tc) ? tc : std::isfinite(tq) ? tq : tb;

            case interpolation::quadratic: return std::isfinite(tq) ? tq : tb;

            case interpolation::bisection:
            default: return tb;
            }
        }

        // attributes
        scalar_t t{0}; ///< line-search step
        scalar_t f{0}; ///< line-search function value
        scalar_t g{0}; ///< line-search gradient
    };

    template <>
    inline enum_map_t<lsearch_step_t::interpolation> enum_string<lsearch_step_t::interpolation>()
    {
        return {
            {lsearch_step_t::interpolation::bisection, "bisection"},
            {lsearch_step_t::interpolation::quadratic, "quadratic"},
            {    lsearch_step_t::interpolation::cubic,     "cubic"}
        };
    }
} // namespace nano
