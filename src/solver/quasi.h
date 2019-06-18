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
    class solver_quasi_t : public solver_t
    {
    public:

        ///
        /// \brief methods to initialize the first approximation of the Hessian's inverse.
        ///
        enum class initialization
        {
            identity,       ///< H0 = I
            scaled,         ///< H0 = I * dg.dot(dx) / dg.dot(dg) - see (2)
        };

        solver_quasi_t();
        json_t config() const override;
        void config(const json_t&) override;
        solver_state_t minimize(const solver_function_t&, const lsearch_t&, const vector_t& x0) const final;

    private:

        virtual void update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const = 0;

        // attributes
        initialization  m_initialization{initialization::identity}; ///<
    };

    ///
    /// \brief Symmetric Rank One (SR1).
    ///
    class solver_quasi_sr1_t final : public solver_quasi_t
    {
    public:

        json_t config() const final;
        void config(const json_t&) final;
        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;

    private:

        // attributes
        scalar_t        m_r{1e-8};      ///< threshold to skip updates when the denominator is too small - see (2)
    };

    ///
    /// \brief Davidon-Fletcher-Powell (DFP).
    ///
    class solver_quasi_dfp_t final : public solver_quasi_t
    {
    public:

        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;
    };

    ///
    /// \brief Broyden-Fletcher-Goldfarb-Shanno (BFGS).
    ///
    class solver_quasi_bfgs_t final : public solver_quasi_t
    {
    public:

        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;
    };

    ///
    /// \brief Hoshino formula (part of Broyden family) for the convex class.
    ///
    class solver_quasi_hoshino_t final : public solver_quasi_t
    {
    public:

        void update(const solver_state_t&, const solver_state_t&, matrix_t&) const final;
    };

    ///
    /// \brief Fletcher switch (SR1 truncated to the convex class) - see (4).
    ///
    class solver_quasi_fletcher_t final : public solver_quasi_t
    {
    public:

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
