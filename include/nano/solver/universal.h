#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief universal gradient methods.
    ///     see "Universal Gradient Methods for Convex Optimization Problems", by Yu. Nesterov, 2013
    ///
    /// NB: the algorithm was designed to minimize a structured convex problem,
    ///     but here it is applied to a (sub-)differentiable convex function directly.
    ///
    /// NB: the original stopping criterion is too loose in practice and it depends on a typically
    ///     unknown distance from the starting point to the optimum - D. instead, these methods
    ///     stop early only for smooth problems with the usual criterion on the magnitude of the gradient.
    ///
    /// NB: the proxy function is the squared euclidean distance: d(x) = 1/2 ||x - x0||^2.
    ///
    /// NB: generally these methods are slow and they depends significantly on the choice of the proxy
    ///     function and sometimes on the initial estimation of the Lipschitz constant - L.
    ///
    /// NB: best results are obtained for smooth functions or when the desired accuracy is low (e.g. 1e-2 to 1e-3).
    ///
    class solver_universal_t : public solver_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_universal_t();
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
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;
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
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;
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
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;
    };
}
