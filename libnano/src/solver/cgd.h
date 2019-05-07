#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief non-linear conjugate gradient descent with line-search.
    ///     see (1) "A survey of nonlinear conjugate gradient methods", by William W. Hager and Hongchao Zhang
    ///     see (2) "Nonlinear Conjugate Gradient Methods", by Yu-Hong Dai
    ///     see (3) "A new conjugate gradient method with guaranteed descent and an efficient line search", by Hager & Zhang
    ///
    class solver_cgd_t : public solver_t
    {
    public:

        solver_cgd_t();
        json_t config() const override;
        void config(const json_t&) override;
        solver_state_t minimize(const solver_function_t&, const lsearch_t&, const vector_t& x0) const final;

    private:

        ///
        /// \brief compute the adjustment factor for the descent direction
        ///
        virtual scalar_t beta(const solver_state_t& prev, const solver_state_t& curr) const = 0;

        // attributes
        scalar_t        m_orthotest{0.1};       ///< orthogonality test
    };

    ///
    /// \brief CGD update parameters (Hager and Zhang, 2005 - see (1)) aka CG_DESCENT
    ///
    class solver_cgd_n_t final : public solver_cgd_t
    {
    public:

        json_t config() const final;
        void config(const json_t&) final;
        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;

    private:

        // attributes
        scalar_t        m_eta{0.01};            ///< see CG_DESCENT
    };

    ///
    /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987 - see (1))
    ///
    class solver_cgd_cd_t final : public solver_cgd_t
    {
    public:

        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Dai and Yuan, 1999 - see (1))
    ///
    class solver_cgd_dy_t final : public solver_cgd_t
    {
    public:

        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Fletcher and Reeves, 1964 - see (1))
    ///
    class solver_cgd_fr_t final : public solver_cgd_t
    {
    public:

        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Hestenes and Stiefel, 1952 - see (1))
    ///
    class solver_cgd_hs_t final : public solver_cgd_t
    {
    public:

        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Liu and Storey, 1991 - see (1))
    ///
    class solver_cgd_ls_t final : public solver_cgd_t
    {
    public:

        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Polak and Ribiere, 1969 - see (1))
    ///
    class solver_cgd_pr_t final : public solver_cgd_t
    {
    public:

        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Dai, 2002 - see (2), page 22)
    ///
    class solver_cgd_dycd_t final : public solver_cgd_t
    {
    public:

        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Dai and Yuan, 2001  - see (2), page 21)
    ///
    class solver_cgd_dyhs_t final : public solver_cgd_t
    {
    public:

        virtual scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };
}
