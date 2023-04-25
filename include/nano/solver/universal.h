#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief universal gradient methods.
///     see "Universal Gradient Methods for Convex Optimization Problems", by Yu. Nesterov, 2013
///
/// NB: the functional constraints (if any) are all ignored.
///
/// NB: the algorithm was designed to minimize a structured convex problem,
///     but here it is applied to a (sub-)differentiable convex function directly.
///
/// NB: the original stopping criterion is too loose in practice and it depends on a typically
///     unknown distance from the starting point to the optimum - D. instead, the iterations are stopped
///     when no significant decrease in the function value in the recent iterations.
///
/// NB: the proxy function is the squared euclidean distance: d(x) = 1/2 ||x - x0||^2.
///
/// NB: generally these methods are slow and they depends significantly on the choice of the proxy
///     function and sometimes on the initial estimation of the Lipschitz constant - L.
///
class NANO_PUBLIC solver_universal_t : public solver_t
{
public:
    ///
    /// \brief default constructor
    ///
    explicit solver_universal_t(string_t id);
};

///
/// \brief universal primal gradient method (PGM).
///     see "Universal Gradient Methods for Convex Optimization Problems", by Yu. Nesterov, 2013
///
class NANO_PUBLIC solver_pgm_t final : public solver_universal_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_pgm_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;
};

///
/// \brief universal dual gradient method (DGM).
///     see "Universal Gradient Methods for Convex Optimization Problems", by Yu. Nesterov, 2013
///
class NANO_PUBLIC solver_dgm_t final : public solver_universal_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_dgm_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;
};

///
/// \brief universal fast gradient method (FGM).
///     see "Universal Gradient Methods for Convex Optimization Problems", by Yu. Nesterov, 2013
///
class NANO_PUBLIC solver_fgm_t final : public solver_universal_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_fgm_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;
};
} // namespace nano
