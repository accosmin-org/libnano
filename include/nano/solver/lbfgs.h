#pragma once

#include <nano/solver/lsearch.h>

namespace nano
{
    ///
    /// \brief limited memory BGFS (l-BGFS).
    ///     see "Updating Quasi-Newton Matrices with Limited Storage", by J. Nocedal, 1980
    ///     see "Numerical Optimization", by J. Nocedal, S. Wright, 2006
    ///
    class NANO_PUBLIC solver_lbfgs_t final : public lsearch_solver_t
    {
    public:

        using solver_t::minimize;

        ///
        /// \brief default constructor
        ///
        solver_lbfgs_t();

        ///
        /// \brief @see lsearch_solver_t
        ///
        solver_state_t iterate(const solver_function_t&, const lsearch_t&, const vector_t& x0) const final;

        ///
        /// \brief change parameters
        ///
        void history(const size_t history) { m_history = history; }

        ///
        /// \brief access functions
        ///
        auto history() const { return m_history.get(); }

    private:

        // attributes
        uparam1_t   m_history{"solver::lbfgs::history", 1, LE, 6, LE, 1000};///<#previous gradients to approximate Hessian^-1
    };
}
