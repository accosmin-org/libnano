#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief universal primal gradient method (PGM).
    ///     see "Universal Gradient Methods for Convex Optimization Problems", by Yu. Nesterov, 2013
    ///
    /// NB: the algorithm was designed to minimize a structured convex problem,
    ///     but here it is applied to a (sub-)differentiable convex function directly.
    ///
    class NANO_PUBLIC solver_pgm_t final : public solver_t
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
    /// NB: the algorithm was designed to minimize a structured convex problem,
    ///     but here it is applied to a (sub-)differentiable convex function directly.
    ///
    class NANO_PUBLIC solver_dgm_t final : public solver_t
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
    /// NB: the algorithm was designed to minimize a structured convex problem,
    ///     but here it is applied to a (sub-)differentiable convex function directly.
    ///
    class NANO_PUBLIC solver_fgm_t final : public solver_t
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
