#pragma once

#include <nano/core/strutil.h>
#include <nano/function.h>

namespace nano
{
    enum class solver_status : int32_t
    {
        max_iters, ///< maximum number of iterations reached without convergence (default)
        converged, ///< convergence criterion reached
        failed,    ///< optimization failed (e.g. line-search failed)
        stopped    ///< user requested stop
    };

    ///
    /// \brief models a state (step) in an unconstrained numerical optimization method:
    ///     * current point (x),
    ///     * function value (f),
    ///     * function gradient (g),
    ///     * constraint equalities (ceq) - the value of equality constraints (if any),
    ///     * constraint inequalities (cineq) - the value of inequality constraints (if any),
    ///     * descent direction (d) - if applicable,
    ///     * line-search step (t) - if applicable.
    ///
    class NANO_PUBLIC solver_state_t
    {
    public:
        ///
        /// \brief default constructor
        ///
        solver_state_t();

        ///
        /// \brief constructor
        ///
        solver_state_t(const function_t&, vector_t x0);

        ///
        /// \brief move to another point.
        ///
        template <typename tvector>
        bool update(const tvector& xx)
        {
            assert(function);
            assert(x.size() == xx.size());
            assert(x.size() == function->size());
            x = xx;
            f = function->vgrad(x, &g);
            update_constraints();
            return valid();
        }

        ///
        /// \brief line-search step along the descent direction of state0.
        /// returns true if the update was successfully.
        ///
        bool update(const solver_state_t& state0, scalar_t t)
        {
            this->t = t;
            return update(state0.x + t * state0.d);
        }

        ///
        /// \brief update the current state, if the given function value is smaller than the current one.
        /// returns true if the update was performed.
        ///
        bool update_if_better(const vector_t& x, scalar_t fx);
        bool update_if_better(const vector_t& x, const vector_t& gx, scalar_t fx);

        ///
        /// \brief update the current state, if the constraints are approximatively improved.
        /// returns true if the update was performed.
        ///
        /// NB: the function value is re-evaluated at the given point if updated, as the given state
        ///     can be a modified function (e.g. penalty, augmented lagrangian).
        ///
        bool update_if_better_constrained(const solver_state_t&, scalar_t epsilon);

        ///
        /// \brief convergence criterion of the gradient: the gradient magnitude relative to the function value.
        ///
        /// NB: only appropriate for smooth problems.
        ///
        scalar_t gradient_test() const;

        ///
        /// \brief convergence criterion of the constraints (if any).
        ///
        scalar_t constraint_test() const;

        ///
        /// \brief check divergence.
        ///
        bool valid() const;

        ///
        /// \brief compute the dot product between the gradient and the descent direction.
        ///
        auto dg() const { return g.dot(d); }

        ///
        /// \brief check if the chosen direction is a descent direction.
        ///
        auto has_descent() const { return dg() < 0; }

        ///
        /// \brief check if the current step satisfies the Armijo condition (sufficient decrease).
        ///
        bool has_armijo(const solver_state_t& state0, scalar_t c1) const
        {
            assert(c1 > 0 && c1 < 1);
            return f <= state0.f + t * c1 * state0.dg();
        }

        ///
        /// \brief check if the current step satisfies the approximate Armijo condition (sufficient decrease).
        ///     see CG_DESCENT
        ///
        bool has_approx_armijo(const solver_state_t& state0, scalar_t epsilon) const { return f <= state0.f + epsilon; }

        ///
        /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature).
        ///
        bool has_wolfe(const solver_state_t& state0, scalar_t c2) const
        {
            assert(c2 > 0 && c2 < 1);
            return dg() >= c2 * state0.dg();
        }

        ///
        /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature).
        ///
        bool has_strong_wolfe(const solver_state_t& state0, scalar_t c2) const
        {
            assert(c2 > 0 && c2 < 1);
            return std::fabs(dg()) <= c2 * std::fabs(state0.dg());
        }

        ///
        /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature).
        ///     see CG_DESCENT
        ///
        bool has_approx_wolfe(const solver_state_t& state0, scalar_t c1, scalar_t c2) const
        {
            assert(0 < c1 && c1 < scalar_t(0.5) && c1 < c2 && c2 < 1);
            return (2.0 * c1 - 1.0) * state0.dg() >= dg() && dg() >= c2 * state0.dg();
        }

        // attributes
        const function_t* function{nullptr};                ///<
        vector_t          x, g, d;                          ///< parameter, gradient, descent direction
        scalar_t          f{0};                             ///< function value
        scalar_t          t{0};                             ///< step size (line-search solvers)
        vector_t          ceq;                              ///< equality constraint values
        vector_t          cineq;                            ///< inequality constraint values
        solver_status     status{solver_status::max_iters}; ///< optimization status
        tensor_size_t     fcalls{0};                        ///< number of function value evaluations so far
        tensor_size_t     gcalls{0};                        ///< number of function gradient evaluations so far
        tensor_size_t     inner_iters{0};                   ///< number of inner iterations so far
        tensor_size_t     outer_iters{0};                   ///< number of outer iterations so far (if constrained)

    private:
        void update_constraints();
    };

    ///
    /// \brief convergence test that checks two consecutive (best) states are close enough.
    ///
    /// NB: appropriate for non-smooth or constrained problems.
    ///
    NANO_PUBLIC bool converged(const solver_state_t& best_state, const solver_state_t& current_state, scalar_t epsilon);

    template <>
    NANO_PUBLIC enum_map_t<solver_status> enum_string<solver_status>();
    NANO_PUBLIC std::ostream& operator<<(std::ostream& os, solver_status);
    NANO_PUBLIC std::ostream& operator<<(std::ostream& os, const solver_state_t&);
    NANO_PUBLIC bool          operator<(const solver_state_t& lhs, const solver_state_t& rhs);
} // namespace nano
