#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief non-linear conjugate gradient descent with line-search.
///
/// see (1) "A survey of nonlinear conjugate gradient methods", by Hager, Zhang
/// see (2) "Nonlinear Conjugate Gradient Methods", by Dai
/// see (3) "A new conjugate gradient method with guaranteed descent and an efficient line search", by Hager, Zhang
/// see (4) "Numerical optimization", Nocedal & Wright, 2nd edition, 2006
///
class NANO_PUBLIC solver_cgd_t : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit solver_cgd_t(string_t id);

    ///
    /// \brief @see solver_t
    ///
    bool has_lsearch() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0, const logger_t&) const override;

private:
    ///
    /// \brief compute the adjustment factor for the descent direction
    ///     from the previous gradient, previous descent direction and current gradient.
    ///
    virtual scalar_t beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const = 0;
};

///
/// \brief CGD update parameters (Hager and Zhang, 2005 - see (1)) aka CG_DESCENT
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_n_t final : public solver_cgd_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_cgd_n_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987 - see (1))
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_cd_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_cd_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (Dai and Yuan, 1999 - see (1))
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_dy_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_dy_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (Fletcher and Reeves, 1964 - see (1))
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_fr_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_fr_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (Hestenes and Stiefel, 1952 - see (1))
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_hs_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_hs_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (Liu and Storey, 1991 - see (1))
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_ls_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_ls_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (Polak and Ribiere, 1969 - see (1))
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_pr_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_pr_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (Dai, 2002 - see (2), page 22)
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_dycd_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_dycd_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (Dai and Yuan, 2001  - see (2), page 21)
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_dyhs_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_dyhs_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};

///
/// \brief CGD update parameters (FR-PR - see (4), formula 5.48)
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_cgd_frpr_t final : public solver_cgd_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_cgd_frpr_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_cgd_t
    ///
    scalar_t beta(const vector_t&, const vector_t&, const vector_t&) const override;
};
} // namespace nano
