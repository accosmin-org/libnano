#include "cgd.h"
#include <nano/numeric.h>

using namespace nano;

template <typename tcgd>
solver_cgd_t<tcgd>::solver_cgd_t() :
    solver_t(1e-4, 1e-1)
{
}

template <typename tcgd>
json_t solver_cgd_t<tcgd>::config() const
{
    json_t json = solver_t::config();
    json["orthotest"] = strcat(m_orthotest, "(0,1)");
    return json;
}

template <typename tcgd>
void solver_cgd_t<tcgd>::config(const json_t& json)
{
    const auto eps = epsilon0<scalar_t>();

    solver_t::config(json);
    nano::from_json_range(json, "orthotest", m_orthotest, eps, 1 - eps);
}

template <typename tcgd>
solver_state_t solver_cgd_t<tcgd>::minimize(const solver_function_t& function, const vector_t& x0) const
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
            const scalar_t beta = tcgd::get(pstate, cstate);
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
        const auto iter_ok = lsearch(cstate);
        if (solver_t::done(function, cstate, iter_ok))
        {
            break;
        }
    }

    return cstate;
}

template class nano::solver_cgd_t<cgd_step_HS>;
template class nano::solver_cgd_t<cgd_step_FR>;
template class nano::solver_cgd_t<cgd_step_PRP>;
template class nano::solver_cgd_t<cgd_step_CD>;
template class nano::solver_cgd_t<cgd_step_LS>;
template class nano::solver_cgd_t<cgd_step_DY>;
template class nano::solver_cgd_t<cgd_step_N>;
template class nano::solver_cgd_t<cgd_step_DYCD>;
template class nano::solver_cgd_t<cgd_step_DYHS>;
