#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief quasi-Newton methods.
    ///     see (1) "Practical methods of optimization", Fletcher, 2nd edition
    ///     see (2) "Numerical optimization", Nocedal & Wright, 2nd edition
    ///     see (3) "Introductory Lectures on Convex Optimization (Applied Optimization)", Nesterov, 2013
    ///     see (4) "A new approach to variable metric algorithms", Fletcher, 1972
    ///
    class NANO_PUBLIC solver_quasi_t : public solver_t
    {
    public:

        using solver_t::minimize;

        ///
        /// \brief methods to initialize the first approximation of the Hessian's inverse.
        ///
        enum class initialization
        {
            identity,       ///< H0 = I
            scaled,         ///< H0 = I * dg.dot(dx) / dg.dot(dg) - see (2)
        };

        ///
        /// \brief default constructor
        ///
        solver_quasi_t();

        ///
        /// \brief @see lsearch_solver_t
        ///
        [[nodiscard]] solver_state_t iterate(const solver_function_t&, const lsearch_t&, const vector_t& x0) const final;

        ///
        /// \brief change parameters
        ///
        void init(const initialization init) { m_initialization = init; }

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto init() const { return m_initialization; }

    private:

        virtual void update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const = 0;

        // attributes
        initialization  m_initialization{initialization::identity}; ///<
    };

    ///
    /// \brief Symmetric Rank One (SR1).
    ///
    class NANO_PUBLIC solver_quasi_sr1_t final : public solver_quasi_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_quasi_sr1_t() = default;

        ///
        /// \brief @see solver_quasi_t
        ///
        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;

        ///
        /// \brief change parameters
        ///
        void r(const scalar_t r) { m_r = r; }

        ///
        /// \brief access functions
        ///
        [[nodiscard]] auto r() const { return m_r.get(); }

    private:

        // attributes
        sparam1_t   m_r{"solver::quasiSR1::r", 0, LT, 1e-8, LT, 1}; ///< threshold to skip updates when the denominator is too small - see (2)
    };

    ///
    /// \brief Davidon-Fletcher-Powell (DFP).
    ///
    class NANO_PUBLIC solver_quasi_dfp_t final : public solver_quasi_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_quasi_dfp_t() = default;

        ///
        /// \brief @see solver_quasi_t
        ///
        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;
    };

    ///
    /// \brief Broyden-Fletcher-Goldfarb-Shanno (BFGS).
    ///
    class NANO_PUBLIC solver_quasi_bfgs_t final : public solver_quasi_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_quasi_bfgs_t() = default;

        ///
        /// \brief @see solver_quasi_t
        ///
        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;
    };

    ///
    /// \brief Hoshino formula (part of Broyden family) for the convex class.
    ///
    class NANO_PUBLIC solver_quasi_hoshino_t final : public solver_quasi_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_quasi_hoshino_t() = default;

        ///
        /// \brief @see solver_quasi_t
        ///
        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;
    };

    ///
    /// \brief Fletcher switch (SR1 truncated to the convex class) - see (4).
    ///
    class NANO_PUBLIC solver_quasi_fletcher_t final : public solver_quasi_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_quasi_fletcher_t() = default;

        ///
        /// \brief @see solver_quasi_t
        ///
        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;
    };

    template <>
    inline enum_map_t<solver_quasi_t::initialization> enum_string<solver_quasi_t::initialization>()
    {
        return
        {
            { solver_quasi_t::initialization::identity,     "identity" },
            { solver_quasi_t::initialization::scaled,       "scaled" }
        };
    }
}
