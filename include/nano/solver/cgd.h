#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief non-linear conjugate gradient descent with line-search.
    ///     see (1) "A survey of nonlinear conjugate gradient methods", by William W. Hager and Hongchao Zhang
    ///     see (2) "Nonlinear Conjugate Gradient Methods", by Yu-Hong Dai
    ///     see (3) "A new conjugate gradient method with guaranteed descent and an efficient line search", by Hager & Zhang
    ///     see (4) "Numerical optimization", Nocedal & Wright, 2nd edition
    ///
    class NANO_PUBLIC solver_cgd_t : public solver_t
    {
    public:

        ///
        /// \brief constructor
        ///
        solver_cgd_t();

        ///
        /// \brief @see solver_t
        ///
        solver_state_t minimize(const function_t&, const vector_t& x0) const final;

    private:

        ///
        /// \brief compute the adjustment factor for the descent direction
        ///
        virtual scalar_t beta(const solver_state_t& prev, const solver_state_t& curr) const = 0;
    };

    ///
    /// \brief CGD update parameters (Hager and Zhang, 2005 - see (1)) aka CG_DESCENT
    ///
    class NANO_PUBLIC solver_cgd_n_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief constructor
        ///
        solver_cgd_n_t();

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987 - see (1))
    ///
    class NANO_PUBLIC solver_cgd_cd_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_cd_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Dai and Yuan, 1999 - see (1))
    ///
    class NANO_PUBLIC solver_cgd_dy_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_dy_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Fletcher and Reeves, 1964 - see (1))
    ///
    class NANO_PUBLIC solver_cgd_fr_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_fr_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Hestenes and Stiefel, 1952 - see (1))
    ///
    class NANO_PUBLIC solver_cgd_hs_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_hs_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Liu and Storey, 1991 - see (1))
    ///
    class NANO_PUBLIC solver_cgd_ls_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_ls_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Polak and Ribiere, 1969 - see (1))
    ///
    class NANO_PUBLIC solver_cgd_pr_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_pr_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Dai, 2002 - see (2), page 22)
    ///
    class NANO_PUBLIC solver_cgd_dycd_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_dycd_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (Dai and Yuan, 2001  - see (2), page 21)
    ///
    class NANO_PUBLIC solver_cgd_dyhs_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_dyhs_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };

    ///
    /// \brief CGD update parameters (FR-PR - see (4), formula 5.48)
    ///
    class NANO_PUBLIC solver_cgd_frpr_t final : public solver_cgd_t
    {
    public:

        ///
        /// \brief default constructor
        ///
        solver_cgd_frpr_t() = default;

        ///
        /// \brief @see solver_cgd_t
        ///
        scalar_t beta(const solver_state_t&, const solver_state_t&) const final;
    };
}
