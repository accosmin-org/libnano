#pragma once

#include <nano/lsearchk.h>

namespace nano
{
///
/// \brief CG_DESCENT:
///
/// see (1) "A new conjugate gradient method with guaranteed descent and an efficient line search",
///         by Hager, Zhang, 2005
///
/// see (2) "Algorithm 851: CG_DESCENT, a Conjugate Gradient Method with Guaranteed Descent",
///         by Hager, Zhang, 2006
///
/// NB: The implementation follows the notation from (1).
///
class NANO_PUBLIC lsearchk_cgdescent_t final : public lsearchk_t
{
public:
    struct params_t;
    struct interval_t;

    ///
    /// \brief constructor
    ///
    lsearchk_cgdescent_t();

    ///
    /// \brief @see lsearchk_t
    ///
    rlsearchk_t clone() const override;

    ///
    /// \brief @see lsearchk_t
    ///
    result_t do_get(const solver_state_t&, const vector_t&, scalar_t, solver_state_t&, const logger_t&) const override;

private:
    bool move(interval_t&, scalar_t step_size, const logger_t&) const;
    void update(interval_t&, const params_t&, const logger_t&) const;
    void updateU(interval_t&, const params_t&, const logger_t&) const;
    void bracket(interval_t&, const params_t&, const logger_t&) const;
};
} // namespace nano
