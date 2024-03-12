#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief methods to initialize the first approximation of the Hessian's inverse.
///
enum class quasi_initialization
{
    identity, ///< H0 = I
    scaled,   ///< H0 = I * dg.dot(dx) / dg.dot(dg) - see (2)
};

template <>
inline enum_map_t<quasi_initialization> enum_string()
{
    return {
        {quasi_initialization::identity, "identity"},
        {  quasi_initialization::scaled,   "scaled"}
    };
}

///
/// \brief quasi-Newton methods.
///
/// see (1) "Practical methods of optimization", Fletcher, 2nd edition
/// see (2) "Numerical optimization", Nocedal & Wright, 2nd edition
/// see (3) "Introductory Lectures on Convex Optimization (Applied Optimization)", Nesterov, 2013
/// see (4) "A new approach to variable metric algorithms", Fletcher, 1972
///
class NANO_PUBLIC solver_quasi_t : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit solver_quasi_t(string_t id);

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;

private:
    virtual void update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const = 0;
};

///
/// \brief Symmetric Rank One (SR1).
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_quasi_sr1_t final : public solver_quasi_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_quasi_sr1_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_quasi_t
    ///
    void update(const solver_state_t&, const solver_state_t&, matrix_t&) const override;
};

///
/// \brief Davidon-Fletcher-Powell (DFP).
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_quasi_dfp_t final : public solver_quasi_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_quasi_dfp_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_quasi_t
    ///
    void update(const solver_state_t&, const solver_state_t&, matrix_t&) const override;
};

///
/// \brief Broyden-Fletcher-Goldfarb-Shanno (BFGS).
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_quasi_bfgs_t final : public solver_quasi_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_quasi_bfgs_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_quasi_t
    ///
    void update(const solver_state_t&, const solver_state_t&, matrix_t&) const override;
};

///
/// \brief Hoshino formula (part of Broyden family) for the convex class.
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_quasi_hoshino_t final : public solver_quasi_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_quasi_hoshino_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_quasi_t
    ///
    void update(const solver_state_t&, const solver_state_t&, matrix_t&) const override;
};

///
/// \brief Fletcher switch (SR1 truncated to the convex class) - see (4).
///
/// NB: the functional constraints (if any) are all ignored.
///
class NANO_PUBLIC solver_quasi_fletcher_t final : public solver_quasi_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_quasi_fletcher_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_quasi_t
    ///
    void update(const solver_state_t&, const solver_state_t&, matrix_t&) const override;
};
} // namespace nano
