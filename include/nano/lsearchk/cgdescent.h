#pragma once

#include <nano/lsearchk.h>

namespace nano
{
///
/// \brief CG_DESCENT:
///     see (1) "A new conjugate gradient method with guaranteed descent and an efficient line search",
///     by William W. Hager & HongChao Zhang, 2005
///
///     see (2) "Algorithm 851: CG_DESCENT, a Conjugate Gradient Method with Guaranteed Descent",
///     by William W. Hager & HongChao Zhang, 2006
///
/// NB: The implementation follows the notation from (1).
///
class NANO_PUBLIC lsearchk_cgdescent_t final : public lsearchk_t
{
public:
    struct interval_t
    {
        interval_t(const solver_state_t& state0, const vector_t& descent, scalar_t step_size, solver_state_t&);

        void updateA() { a = {c, descent, step_size}; }

        void updateB() { b = {c, descent, step_size}; }

        bool done(scalar_t c1, scalar_t c2, scalar_t epsilonk, bool bracketed = true) const;

        const solver_state_t& state0;    ///< original point
        const vector_t&       descent;   ///< descent direction
        scalar_t              step_size; ///< step size of the tentative point
        solver_state_t&       c;         ///< tentative point
        lsearch_step_t        a;         ///< lower bounds of the bracketing interval
        lsearch_step_t        b;         ///< upper bounds of the bracketing interval
    };

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
    result_t do_get(const solver_state_t&, const vector_t&, scalar_t, solver_state_t&) const override;

private:
    void move(interval_t&, scalar_t step_size) const;
    void update(interval_t&, scalar_t epsilonk, scalar_t theta, int max_iterations) const;
    void updateU(interval_t&, scalar_t epsilonk, scalar_t theta, int max_iterations) const;
    void bracket(interval_t&, scalar_t ro, scalar_t epsilonk, scalar_t theta, int max_iterations) const;
};
} // namespace nano
