#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief non-linear conjugate gradient descent with line-search.
    ///      see (1) "A survey of nonlinear conjugate gradient methods", by William W. Hager and Hongchao Zhang
    ///      see (2) "Nonlinear Conjugate Gradient Methods", by Yu-Hong Dai
    ///
    template <typename tcgd>
    class solver_cgd_t final : public solver_t
    {
    public:

        solver_cgd_t();
        json_t config() const final;
        void config(const json_t&) final;
        solver_state_t minimize(const solver_function_t&, const lsearch_t&, const vector_t& x0) const final;

    private:

        // attributes
        scalar_t        m_orthotest{0.1};       ///< orthogonality test
    };

    struct cgd_step_N;
    struct cgd_step_CD;
    struct cgd_step_DY;
    struct cgd_step_FR;
    struct cgd_step_HS;
    struct cgd_step_LS;
    struct cgd_step_PRP;
    struct cgd_step_DYCD;
    struct cgd_step_DYHS;

    // create various CGD algorithms
    using solver_cgd_n_t = solver_cgd_t<cgd_step_N>;
    using solver_cgd_cd_t = solver_cgd_t<cgd_step_CD>;
    using solver_cgd_dy_t = solver_cgd_t<cgd_step_DY>;
    using solver_cgd_fr_t = solver_cgd_t<cgd_step_FR>;
    using solver_cgd_hs_t = solver_cgd_t<cgd_step_HS>;
    using solver_cgd_ls_t = solver_cgd_t<cgd_step_LS>;
    using solver_cgd_prp_t = solver_cgd_t<cgd_step_PRP>;
    using solver_cgd_dycd_t = solver_cgd_t<cgd_step_DYCD>;
    using solver_cgd_dyhs_t = solver_cgd_t<cgd_step_DYHS>;
}
