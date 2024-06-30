#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief interface for primal-dual subgradient methods.
///
/// see "Primal-dual subgradient methods", by Y. Nesterov, 2009
///
/// NB: the functional constraints (if any) are all ignored.
/// NB: the prox-function is the Euclidean norm.
/// NB: the iterations are stopped when the gap is smaller than epsilon.
/// NB: the algorithm is sensitive to the estimated distance between the initial point and the optimum.
///
class NANO_PUBLIC solver_pdsgm_t : public solver_t
{
public:
    struct model_t;

    ///
    /// \brief default constructor
    ///
    explicit solver_pdsgm_t(string_t id);

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;

private:
    virtual std::tuple<scalar_t, scalar_t> update(const model_t& model, const vector_t& gx) const = 0;
};

///
/// \brief simple dual averages (SDA) variation of primal-dual subgradient methods.
///
class NANO_PUBLIC solver_sda_t final : public solver_pdsgm_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_sda_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

private:
    std::tuple<scalar_t, scalar_t> update(const model_t& model, const vector_t& gx) const override;
};

///
/// \brief weighted dual averages (WDA) variation of primal-dual subgradient methods.
///
class NANO_PUBLIC solver_wda_t final : public solver_pdsgm_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_wda_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

private:
    std::tuple<scalar_t, scalar_t> update(const model_t& model, const vector_t& gx) const override;
};
} // namespace nano
