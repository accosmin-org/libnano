#pragma once

#include <nano/arch.h>
#include <nano/solver/state.h>

namespace nano
{
    ///
    /// \brief interpolation method using the information at two line-search step trials.
    ///
    enum class interpolation_type
    {
        bisection, ///<
        quadratic, ///<
        cubic      ///<
    };

    ///
    /// \brief line-search step function:
    ///     phi(t) = f(x + t * d), f - the function to minimize and d - the descent direction.
    ///
    class NANO_PUBLIC lsearch_step_t
    {
    public:
        ///
        /// \brief constructor
        ///
        lsearch_step_t(scalar_t t, scalar_t f, scalar_t dg);
        lsearch_step_t(const solver_state_t&, const vector_t& descent, scalar_t step_size);

        ///
        /// \brief cubic interpolation of two line-search steps.
        ///     fit cubic: q(x) = a*x^3 + b*x^2 + c*x + d
        ///         given: q(u) = fu, q'(u) = gu
        ///         given: q(v) = fv, q'(v) = gv
        ///     minimizer: solution of 3*a*x^2 + 2*b*x + c = 0
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        static scalar_t cubic(const lsearch_step_t& u, const lsearch_step_t& v);

        ///
        /// \brief quadratic interpolation of two line-search steps.
        ///     fit quadratic: q(x) = a*x^2 + b*x + c
        ///         given: q(u) = fu, q'(u) = gu
        ///         given: q(v) = fv
        ///     minimizer: -b/2a
        ///
        static scalar_t quadratic(const lsearch_step_t& u, const lsearch_step_t& v, bool* convexity = nullptr);

        ///
        /// \brief secant interpolation of two line-search steps.
        ///     fit quadratic: q(x) = a*x^2 + b*x + c
        ///         given: q'(u) = gu
        ///         given: q'(v) = gv
        ///     minimizer: -b/2a
        ///
        static scalar_t secant(const lsearch_step_t& u, const lsearch_step_t& v);

        ///
        /// \brief bisection interpolation of two line-search steps.
        ///
        static scalar_t bisection(const lsearch_step_t& u, const lsearch_step_t& v);

        ///
        /// \brief interpolation of two line-search steps.
        ///     first try a cubic interpolation, then a quadratic interpolation and finally do bisection
        ///         until the interpolated point is valid.
        ///
        static scalar_t interpolate(const lsearch_step_t& u, const lsearch_step_t& v, interpolation_type);

        // attributes
        scalar_t t{0}; ///< line-search step
        scalar_t f{0}; ///< line-search function value
        scalar_t g{0}; ///< line-search dot product gradient and descent direction
    };

    template <>
    inline enum_map_t<interpolation_type> enum_string<interpolation_type>()
    {
        return {
            {interpolation_type::bisection, "bisection"},
            {interpolation_type::quadratic, "quadratic"},
            {    interpolation_type::cubic,     "cubic"}
        };
    }
} // namespace nano
