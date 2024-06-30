#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief accelerated subgradient algorithms (ASGA).
///
/// see "Accelerated first-order methods for large-scale convex minimization", by M. Ahookhosh, 2016
///
/// NB: the algorithm was designed to minimize a structured problem,
///     but here it is applied to a sub-differentiable function directly.
///
/// NB: the default parameters are adapted to obtain as precise solutions as possible
///     and they are quite different from the original paper.
///
/// NB: the estimation of the initial lipschitz value "L0" from the original Matlab code
///     is not working properly in general, while the default 1.0 is working reasonably well overall.
///
/// NB: the algorithm is quite sensitive to the optimum parameter values so that an accurate solution
///     cannot be obtain in all cases even after a large number of iterations.
///
class NANO_PUBLIC solver_asga_t : public solver_t
{
public:
    ///
    /// \brief default constructor
    ///
    explicit solver_asga_t(string_t id);
};

///
/// \brief accelerated subgradient algorithm (ASGA-2).
///
class NANO_PUBLIC solver_asga2_t final : public solver_asga_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_asga2_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const final;
};

///
/// \brief accelerated subgradient algorithm (ASGA-4).
///
class NANO_PUBLIC solver_asga4_t final : public solver_asga_t
{
public:
    ///
    /// \brief default constructor
    ///
    solver_asga4_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const final;
};
} // namespace nano
