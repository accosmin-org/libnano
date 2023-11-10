#pragma once

#include <nano/function.h>
#include <nano/solver/status.h>

namespace nano
{
///
/// \brief models a state (step) in a numerical optimization method.
///
/// NB: it handles both smooth and non-smooth with or without constraints.
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
    /// \brief move to another point and returns true if the new point is valid.
    ///
    template <typename tvector>
    bool update(const tvector& x)
    {
        assert(m_function);
        assert(x.size() == m_x.size());
        assert(x.size() == m_function->size());
        m_x  = x;
        m_fx = m_function->vgrad(m_x, m_gx);
        update_calls();
        update_constraints();
        return valid();
    }

    ///
    /// \brief update the number of function value and gradient evaluations.
    ///
    void update_calls();

    ///
    /// \brief try to update the current state and
    ///     returns true if the given function value is smaller than the current one.
    ///
    /// NB: this is usually called by non-monotonic solvers (e.g. for non-smooth optimization problems).
    ///
    bool update_if_better(const vector_t& x, scalar_t fx);
    bool update_if_better(const vector_t& x, const vector_t& gx, scalar_t fx);

    ///
    /// \brief try to update the current state and
    ///     returns true if the constraints are approximatively improved.
    ///
    /// NB: the function value is re-evaluated at the given point if updated, as the given state
    ///     can be a modified function (e.g. penalty, augmented lagrangian).
    ///
    bool update_if_better_constrained(const solver_state_t&, scalar_t epsilon);

    ///
    /// \brief convergence criterion of the function value:
    ///     improvement in function value and parameter in the most recent updates.
    ///
    /// NB: appropriate for non-monotonic solvers (usually non-smooth problems) that call `update_if_better`.
    ///
    scalar_t value_test(tensor_size_t patience) const;

    ///
    /// \brief convergence criterion of the gradient: the gradient magnitude relative to the function value.
    ///
    /// NB: only appropriate for smooth problems.
    ///
    scalar_t gradient_test() const;
    scalar_t gradient_test(vector_cmap_t gx) const;

    ///
    /// \brief convergence criterion of the constraints (if any).
    ///
    scalar_t constraint_test() const;

    ///
    /// \brief returns true if the current state is valid (e.g. no divergence is detected).
    ///
    bool valid() const;

    ///
    /// \brief returns the dot product between the gradient and the descent direction.
    ///
    /// NB: only appropriate for smooth problems.
    ///
    scalar_t dg(const vector_t& descent) const { return m_gx.dot(descent); }

    ///
    /// \brief returns true if the chosen direction is a descent direction.
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_descent(const vector_t& descent) const { return dg(descent) < 0.0; }

    ///
    /// \brief check if the current step satisfies the Armijo condition (sufficient decrease).
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_armijo(const solver_state_t& origin, const vector_t& descent, scalar_t step_size, scalar_t c1) const;

    ///
    /// \brief check if the current step satisfies the approximate Armijo condition (sufficient decrease).
    ///     see CG_DESCENT
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_approx_armijo(const solver_state_t& origin, scalar_t epsilon) const;

    ///
    /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature).
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_wolfe(const solver_state_t& origin, const vector_t& descent, scalar_t c2) const;

    ///
    /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature).
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_strong_wolfe(const solver_state_t& origin, const vector_t& descent, scalar_t c2) const;

    ///
    /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature).
    ///     see CG_DESCENT
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_approx_wolfe(const solver_state_t& origin, const vector_t& descent, scalar_t c1, scalar_t c2) const;

    ///
    /// \brief set the optimization status.
    ///
    void status(solver_status);

    ///
    /// \brief returns the (sub-)gradient.
    ///
    scalar_t fx() const { return m_fx; }

    ///
    /// \brief returns the current optimum parameter value.
    ///
    const vector_t& x() const { return m_x; }

    ///
    /// \brief returns the (sub-)gradient.
    ///
    const vector_t& gx() const { return m_gx; }

    ///
    /// \brief returns the number of function evaluation calls registered so far.
    ///
    tensor_size_t fcalls() const { return m_fcalls; }

    ///
    /// \brief returns the number of function gradient calls registered so far.
    ///
    tensor_size_t gcalls() const { return m_gcalls; }

    ///
    /// \brief returns the optimization status.
    ///
    solver_status status() const { return m_status; }

    ///
    /// \brief returns the function to minimize.
    ///
    const function_t& function() const
    {
        assert(m_function != nullptr);
        return *m_function;
    }

    ///
    /// \brief returns the values of the equality constraints.
    ///
    const vector_t& ceq() const { return m_ceq; }

    ///
    /// \brief returns the values of the inequality constraints.
    ///
    const vector_t& cineq() const { return m_cineq; }

private:
    void update_constraints();

    using scalars_t = std::vector<scalar_t>;

    // attributes
    const function_t* m_function{nullptr};                ///<
    vector_t          m_x;                                ///< parameter
    vector_t          m_gx;                               ///< gradient
    scalar_t          m_fx{0};                            ///< function value
    vector_t          m_ceq;                              ///< equality constraint values
    vector_t          m_cineq;                            ///< inequality constraint values
    solver_status     m_status{solver_status::max_iters}; ///< optimization status
    tensor_size_t     m_fcalls{0};                        ///< number of function value evaluations so far
    tensor_size_t     m_gcalls{0};                        ///< number of function gradient evaluations so far
    scalars_t         m_history_df;                       ///< recent improvements of the function value
    scalars_t         m_history_dx;                       ///< recent improvements of the parameter
};

///
/// \brief pretty print the given solver state.
///
NANO_PUBLIC std::ostream& operator<<(std::ostream& os, const solver_state_t&);

///
/// \brief convergence test that checks two consecutive (best) states are close enough.
///
/// NB: appropriate for non-smooth or constrained problems.
///
NANO_PUBLIC bool converged(const solver_state_t& best_state, const solver_state_t& current_state, scalar_t epsilon);
} // namespace nano
