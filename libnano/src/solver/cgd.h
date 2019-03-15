#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief non-linear conjugate gradient descent with line-search.
    ///
    template <typename tcgd>
    class solver_cgd_t final : public solver_t
    {
    public:

        solver_cgd_t();
        json_t config() const final;
        void config(const json_t&) final;
        solver_state_t minimize(const solver_function_t&, const vector_t& x0) const final;

    private:

        // attributes
        scalar_t        m_orthotest{0.1};       ///< orthogonality test
    };

    ///
    /// these variations have been implemented following:
    ///      (1) "A survey of nonlinear conjugate gradient methods"
    ///      by William W. Hager and Hongchao Zhang
    ///
    /// and
    ///      (2) "Nonlinear Conjugate Gradient Methods"
    ///      by Yu-Hong Dai

    ///
    /// \brief CGD update parameters (Hestenes and Stiefel, 1952 - see (1))
    ///
    struct cgd_step_HS
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  curr.g.dot(curr.g - prev.g) /
                    prev.d.dot(curr.g - prev.g);
        }
    };

    ///
    /// \brief CGD update parameters (Fletcher and Reeves, 1964 - see (1))
    ///
    struct cgd_step_FR
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  curr.g.squaredNorm() /
                    prev.g.squaredNorm();
        }
    };

    ///
    /// \brief CGD update parameters (Polak and Ribiere, 1969 - see (1))
    ///
    struct cgd_step_PRP
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  std::max(scalar_t(0),            // PRP(+)
                    curr.g.dot(curr.g - prev.g) /
                    prev.g.squaredNorm());
        }
    };

    ///
    /// \brief CGD update parameters (Fletcher - Conjugate Descent, 1987 - see (1))
    ///
    struct cgd_step_CD
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return -curr.g.squaredNorm() /
                    prev.d.dot(prev.g);
        }
    };

    ///
    /// \brief CGD update parameters (Liu and Storey, 1991 - see (1))
    ///
    struct cgd_step_LS
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return -curr.g.dot(curr.g - prev.g) /
                    prev.d.dot(prev.g);
        }
    };

    ///
    /// \brief CGD update parameters (Dai and Yuan, 1999 - see (1))
    ///
    struct cgd_step_DY
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  curr.g.squaredNorm() /
                    prev.d.dot(curr.g - prev.g);
        }
    };

    ///
    /// \brief CGD update parameters (Hager and Zhang, 2005 - see (1)) aka CG_DESCENT
    ///
    struct cgd_step_N
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            const auto y = curr.g - prev.g;
            const auto div = +1 / prev.d.dot(y);

            const auto pd2 = prev.d.lpNorm<2>();
            const auto pg2 = prev.g.lpNorm<2>();
            const auto eta = -1 / (pd2 * std::min(scalar_t(0.01), pg2));

            // N+ (see modification in
            //      "A NEW CONJUGATE GRADIENT METHOD WITH GUARANTEED DESCENT AND AN EFFICIENT LINE SEARCH")
            return  std::max(eta,
                    div * (y - 2 * prev.d * y.squaredNorm() * div).dot(curr.g));
        }
    };

    ///
    /// \brief CGD update parameters (Dai and Yuan, 2001  - see (2), page 21)
    ///
    struct cgd_step_DYHS
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            const auto dy = cgd_step_DY::get(prev, curr);
            const auto hs = cgd_step_HS::get(prev, curr);

            return std::max(scalar_t(0), std::min(dy, hs));
        }
    };

    ///
    /// \brief CGD update parameters (Dai, 2002 - see (2), page 22)
    ///
    struct cgd_step_DYCD
    {
        static scalar_t get(const solver_state_t& prev, const solver_state_t& curr)
        {
            return  curr.g.squaredNorm() /
                    std::max(prev.d.dot(curr.g - prev.g), -prev.d.dot(prev.g));
        }
    };

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
