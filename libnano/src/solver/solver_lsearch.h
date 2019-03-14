#pragma once

#include <nano/solver.h>

namespace nano
{
    ///
    /// \brief base class for line-search based solvers.
    ///
    class solver_lsearch_t : public solver_t
    {
    public:

        solver_lsearch_t(const scalar_t c1, const scalar_t c2);
        void to_json(json_t&) const override;
        void from_json(const json_t&) override;

    protected:

        // attributes
        scalar_t        m_c1{static_cast<scalar_t>(1e-4)};
        scalar_t        m_c2{static_cast<scalar_t>(1e-1)};
        scalar_t        m_orthotest{static_cast<scalar_t>(0.1)};    ///< orthogonality test
    };

    template <typename tcgd_update>
    class solver_cgd_t final : public solver_cgd_base_t
    {
    public:

