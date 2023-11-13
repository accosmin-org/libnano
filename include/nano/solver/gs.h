#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief gradient sampling with line-search.
///
/// see (1) "A robust gradient sampling algorithm for nonsmooth, nonconvex optimization", by Burke, Lewis, Overton, 2005
/// see (2) "Convergence of the gradient sampling algorithm for nonsmooth nonconvex optimization", by Kiwiel, 2007
/// see (3) "The gradient sampling methodology", by Burke, Curtis, Lewis, Overton, 2018
/// see (4) "Two numerical methods for optimizing matrix stability", by Burke, Lewis, Overton, 2002
///
/// NB: the implementation follows the notation from (3), uses the line-search idea from (4) and handles
///     failures in the line-search procedure like in (1) (aka reduce the sampling radius).
///
class NANO_PUBLIC solver_gs_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit solver_gs_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;
};
} // namespace nano
