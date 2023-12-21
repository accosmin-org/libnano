#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief fast proximal bundle algorithm (FPBA).
///
/// see "Proximal bundle algorithms for nonsmooth convex optimization via fast gradient smooth methods", by Ouorou, 2020
///

namespace proximal
{
struct sequence1_t;
struct sequence2_t;

struct fpba1_type_id_t;
struct fpba2_type_id_t;
} // namespace proximal

///
/// \brief base class for fast proximal bundle solvers.
///
template <typename tsequence, typename ttype_id>
class NANO_PUBLIC base_solver_fpba_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit base_solver_fpba_t();

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
/// \brief FPBA1 from (1).
///
using solver_fpba1_t = base_solver_fpba_t<proximal::sequence1_t, proximal::fpba1_type_id_t>;

///
/// \brief FPBA2 from (2).
///
using solver_fpba2_t = base_solver_fpba_t<proximal::sequence2_t, proximal::fpba2_type_id_t>;
} // namespace nano
