#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief gradient sampling methods.
///
/// see (1) "A robust gradient sampling algorithm for nonsmooth, nonconvex optimization", by Burke, Lewis, Overton, 2005
/// see (2) "Convergence of the gradient sampling algorithm for nonsmooth nonconvex optimization", by Kiwiel, 2007
/// see (3) "The gradient sampling methodology", by Burke, Curtis, Lewis, Overton, 2018
/// see (4) "Two numerical methods for optimizing matrix stability", by Burke, Lewis, Overton, 2002
/// see (5) "An adaptive gradient sampling algorithm for nonsmooth optimization", by Curtis, Quez, 2013
/// see (6) "On the differentiability check in gradient sampling methpds", by Helou, Santos, Simeos, 2016
///
/// NB: particularly useful for minimizing non-smooth (convex) problems.
/// NB: strong theoretical guarantees with a practical and theoretically-motivated stopping criterion.
///
/// NB: the implementation follows the notation from (6), in particular:
///     - the line-search is performed with perturbation (P variation) and
///     - the descent direction is non-normalized (nN variation).
///
/// NB: additionally the line-search implementation use the idea from (4)
///     to handle functions that are non-Lipschitz locally.
///

namespace gs
{
struct fixed_sampler_t;
struct adaptive_sampler_t;

class identity_preconditioner_t;
class lbfgs_preconditioner_t;

struct gs_type_id_t;
struct ags_type_id_t;
struct gs_lbfgs_type_id_t;
struct ags_lbfgs_type_id_t;
} // namespace gs

///
/// \brief base class for gradient sampling solvers.
///
template <typename tsampler, typename tpreconditioner, typename ttype_id>
class NANO_PUBLIC base_solver_gs_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit base_solver_gs_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0) const override;
};

///
/// \brief gradient sampling: version P-nNGS from (6).
///
using solver_gs_t = base_solver_gs_t<gs::fixed_sampler_t, gs::identity_preconditioner_t, gs::gs_type_id_t>;

///
/// \brief gradient sampling with LBFGS-like search direction:
///     version P-nNGS from (6) + LBFGS preconditioner from (5).
///
using solver_gs_lbfgs_t = base_solver_gs_t<gs::fixed_sampler_t, gs::lbfgs_preconditioner_t, gs::gs_lbfgs_type_id_t>;

///
/// \brief adaptive gradient sampling: version P-nNGS from (6) + AGS sampling from (5).
///
using solver_ags_t = base_solver_gs_t<gs::adaptive_sampler_t, gs::identity_preconditioner_t, gs::ags_type_id_t>;

///
/// \brief adaptive gradient sampling with LBFGS-like search direction:
///     version P-nNGS from (6) + AGS sampling from (5) + LBFGS preconditioner from (5).
///
using solver_ags_lbfgs_t =
    base_solver_gs_t<gs::adaptive_sampler_t, gs::lbfgs_preconditioner_t, gs::ags_lbfgs_type_id_t>;
} // namespace nano
