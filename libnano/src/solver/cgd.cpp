#include "cgd.h"
#include <nano/numeric.h>

using namespace nano;

solver_cgd_t::solver_cgd_t() :
    solver_t(1e-4, 1e-1)
{
}

json_t solver_cgd_t::config() const
{
    json_t json = solver_t::config();
    json["orthotest"] = strcat(m_orthotest, "(0,1)");
    return json;
}

void solver_cgd_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    solver_t::config(json);
    nano::from_json_range(json, "orthotest", m_orthotest, eps, 1 - eps);
}

solver_state_t solver_cgd_t::minimize(const solver_function_t& function, const lsearch_t& lsearch,
    const vector_t& x0) const
{
    auto cstate = solver_state_t{function, x0};
    auto pstate = cstate;
    log(cstate);

    for (int i = 0; i < max_iterations(); ++ i)
    {
        // descent direction
        if (i == 0)
        {
            cstate.d = -cstate.g;
        }
        else
        {
            const auto beta = this->beta(pstate, cstate);
            cstate.d = -cstate.g + beta * pstate.d;

            // restart:
            //  - if not a descent direction
            //  - or two consecutive gradients far from being orthogonal
            //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.124-125)
            if (!cstate.has_descent())
            {
                cstate.d = -cstate.g;
            }
            else if (std::fabs(cstate.g.dot(pstate.g)) >= m_orthotest * cstate.g.dot(cstate.g))
            {
                cstate.d = -cstate.g;
            }
        }

        // line-search
        pstate = cstate;
        const auto iter_ok = lsearch.get(cstate);
        if (solver_t::done(function, cstate, iter_ok))
        {
            break;
        }
    }

    return cstate;
}

scalar_t solver_cgd_hs_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return  curr.g.dot(curr.g - prev.g) /
            prev.d.dot(curr.g - prev.g);
}

scalar_t solver_cgd_fr_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return  curr.g.squaredNorm() /
            prev.g.squaredNorm();
}

scalar_t solver_cgd_prp_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return  std::max(scalar_t(0),            // PRP(+)
            curr.g.dot(curr.g - prev.g) /
            prev.g.squaredNorm());
}

scalar_t solver_cgd_cd_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return -curr.g.squaredNorm() /
            prev.d.dot(prev.g);
}

scalar_t solver_cgd_ls_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return -curr.g.dot(curr.g - prev.g) /
            prev.d.dot(prev.g);
}

scalar_t solver_cgd_dy_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return  curr.g.squaredNorm() /
            prev.d.dot(curr.g - prev.g);
}

json_t solver_cgd_n_t::config() const
{
    json_t json = solver_cgd_t::config();
    json["eta"] = strcat(m_eta, "(0,inf)");
    return json;
}

void solver_cgd_n_t::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    solver_cgd_t::config(json);
    nano::from_json_range(json, "eta", m_eta, eps, 1 / eps);
}

scalar_t solver_cgd_n_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    const auto y = curr.g - prev.g;
    const auto div = +1 / prev.d.dot(y);

    const auto pd2 = prev.d.lpNorm<2>();
    const auto pg2 = prev.g.lpNorm<2>();
    const auto eta = -1 / (pd2 * std::min(m_eta, pg2));

    // N+ (see modification in
    //      "A NEW CONJUGATE GRADIENT METHOD WITH GUARANTEED DESCENT AND AN EFFICIENT LINE SEARCH")
    return  std::max(eta,
            div * (y - 2 * prev.d * y.squaredNorm() * div).dot(curr.g));
}

scalar_t solver_cgd_dyhs_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    const auto dy = curr.g.squaredNorm() / prev.d.dot(curr.g - prev.g);
    const auto hs = curr.g.dot(curr.g - prev.g) / prev.d.dot(curr.g - prev.g);

    return std::max(scalar_t(0), std::min(dy, hs));
}

scalar_t solver_cgd_dycd_t::beta(const solver_state_t& prev, const solver_state_t& curr) const
{
    return  curr.g.squaredNorm() /
            std::max(prev.d.dot(curr.g - prev.g), -prev.d.dot(prev.g));
}
